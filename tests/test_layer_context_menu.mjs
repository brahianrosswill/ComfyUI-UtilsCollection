import assert from "node:assert/strict";
import test from "node:test";

import { buildLayerContextActions } from "../web/layer_context_menu.js";
import {
  frontmostLayerAtPoint,
  pointInConvexQuad,
} from "../web/placement_geometry.js";

test("convex quad hit testing includes transformed interior and edges", () => {
  const quad = [
    { x: 2, y: 0 },
    { x: 5, y: 2 },
    { x: 3, y: 5 },
    { x: 0, y: 3 },
  ];
  assert.equal(pointInConvexQuad({ x: 2.5, y: 2.5 }, quad), true);
  assert.equal(pointInConvexQuad({ x: 2, y: 0 }, quad), true);
  assert.equal(pointInConvexQuad({ x: 5, y: 5 }, quad), false);
});

test("frontmost hit testing ignores excluded overlapping layers", () => {
  const layers = ["back", "middle", "front"];
  const quad = [
    { x: 0, y: 0 },
    { x: 4, y: 0 },
    { x: 4, y: 4 },
    { x: 0, y: 4 },
  ];
  assert.equal(
    frontmostLayerAtPoint(
      layers,
      { x: 2, y: 2 },
      () => quad,
      (key) => key !== "front",
    ),
    "middle",
  );
  assert.equal(
    frontmostLayerAtPoint(
      layers,
      { x: 8, y: 8 },
      () => quad,
      () => true,
    ),
    null,
  );
});

test("frontmost hit testing passes through locked overlapping layers", () => {
  const layers = ["back", "front"];
  const placements = {
    back: { included: true, locked: false },
    front: { included: true, locked: true },
  };
  const quad = [
    { x: 0, y: 0 },
    { x: 4, y: 0 },
    { x: 4, y: 4 },
    { x: 0, y: 4 },
  ];
  assert.equal(
    frontmostLayerAtPoint(
      layers,
      { x: 2, y: 2 },
      () => quad,
      (key) => placements[key].included !== false && placements[key].locked !== true,
    ),
    "back",
  );
});

test("context actions expose state and boundary availability", () => {
  const invoked = [];
  const callback = (name) => () => invoked.push(name);
  const actions = buildLayerContextActions({
    index: 0,
    count: 3,
    placement: { flip_horizontal: true, flip_vertical: false },
    warpActive: true,
    rotateActive: false,
    moveBack: callback("moveBack"),
    moveForward: callback("moveForward"),
    sendToBack: callback("sendToBack"),
    bringToFront: callback("bringToFront"),
    flipHorizontal: callback("flipHorizontal"),
    flipVertical: callback("flipVertical"),
    toggleWarp: callback("toggleWarp"),
    toggleRotate: callback("toggleRotate"),
    toggleLock: callback("toggleLock"),
    exclude: callback("exclude"),
    reset: callback("reset"),
  }).filter((item) => !item.separator);

  assert.equal(actions.find((item) => item.label === "Move Back").disabled, true);
  assert.equal(actions.find((item) => item.label === "Move Forward").disabled, false);
  assert.equal(actions.find((item) => item.label === "Flip H").checked, true);
  assert.equal(actions.find((item) => item.label === "Flip V").checked, false);
  assert.equal(actions.find((item) => item.label === "Warp").checked, true);
  assert.equal(actions.find((item) => item.label === "Rotate").checked, false);
  assert.equal(actions.find((item) => item.label === "Lock").checked, false);
  actions.find((item) => item.label === "Move Forward").callback();
  actions.find((item) => item.label === "Bring to Front").callback();
  actions.find((item) => item.label === "Exclude").callback();
  assert.deepEqual(invoked, ["moveForward", "bringToFront", "exclude"]);
});

test("locked context state keeps ordering and inclusion available while disabling transforms", () => {
  const noop = () => {};
  const actions = buildLayerContextActions({
    index: 1, count: 3, placement: { locked: true }, warpActive: false, rotateActive: false,
    moveBack: noop, moveForward: noop, sendToBack: noop, bringToFront: noop,
    flipHorizontal: noop, flipVertical: noop, toggleWarp: noop, toggleRotate: noop,
    toggleLock: noop, exclude: noop, reset: noop,
  }).filter((item) => !item.separator);
  assert.equal(actions.find((item) => item.label === "Unlock").checked, true);
  assert.equal(actions.find((item) => item.label === "Exclude").disabled, false);
  assert.equal(actions.find((item) => item.label === "Move Back").disabled, false);
  for (const label of ["Flip H", "Flip V", "Warp", "Rotate", "Reset"]) {
    assert.equal(actions.find((item) => item.label === label).disabled, true);
  }
});
