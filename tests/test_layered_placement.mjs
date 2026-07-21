import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_PLACEMENT,
  drawRect,
  moveRect,
  parsePlacementData,
  placementToRect,
  rectToPlacement,
  resizeRectFromDelta,
  serializePlacementData,
} from "../web/placement_geometry.js";

const close = (actual, expected, tolerance = 1e-9) => assert.ok(
  Math.abs(actual - expected) <= tolerance,
  `${actual} was not within ${tolerance} of ${expected}`,
);

test("landscape, portrait, and square map logical axes correctly", () => {
  const landscape = placementToRect(200, 100, 1, {
    scale: 0.5,
    long_axis_shift: 1,
    short_axis_shift: -1,
  });
  assert.deepEqual(landscape, { x: 150, y: 0, width: 50, height: 50 });

  const portrait = placementToRect(100, 200, 1, {
    scale: 0.5,
    long_axis_shift: 1,
    short_axis_shift: -1,
  });
  assert.deepEqual(portrait, { x: 0, y: 150, width: 50, height: 50 });

  const square = placementToRect(100, 100, 2, {
    scale: 0.5,
    long_axis_shift: 1,
    short_axis_shift: -1,
  });
  assert.deepEqual(square, { x: 50, y: 0, width: 50, height: 25 });
});

test("oversized placement uses negative travel and round trips", () => {
  const placement = { scale: 2, long_axis_shift: 0.6, short_axis_shift: -0.4 };
  const rect = placementToRect(160, 100, 0.5, placement);
  assert.ok(rect.height > 100);
  assert.ok(rect.y < 0);
  const roundTrip = rectToPlacement(160, 100, rect, placement);
  close(roundTrip.scale, placement.scale);
  const rebuilt = placementToRect(160, 100, 0.5, roundTrip);
  close(rebuilt.x, rect.x);
  close(rebuilt.y, rect.y);
});

test("legacy zero-travel placement migrates to an equivalent center", () => {
  const prior = { scale: 1, long_axis_shift: 0.75, short_axis_shift: -0.25 };
  const rect = placementToRect(100, 200, 1, prior);
  const result = rectToPlacement(100, 200, rect, prior);
  assert.deepEqual(placementToRect(100, 200, 1, result), rect);
});

test("drawing in opposite directions produces the same normalized bounds", () => {
  const forward = drawRect({ x: 10, y: 20 }, { x: 110, y: 120 }, 2);
  const reverse = drawRect({ x: 110, y: 120 }, { x: 10, y: 20 }, 2);
  assert.deepEqual(forward, reverse);
  close(forward.width / forward.height, 2);
});

test("moving clamps both contained and oversized rectangles", () => {
  assert.deepEqual(
    moveRect(100, 100, { x: 25, y: 25, width: 50, height: 50 }, 100, -100),
    { x: 50, y: 0, width: 50, height: 50 },
  );
  assert.deepEqual(
    moveRect(100, 100, { x: -50, y: -20, width: 200, height: 140 }, 100, -100),
    { x: 0, y: -40, width: 200, height: 140 },
  );
  assert.deepEqual(
    moveRect(100, 100, { x: -50, y: -20, width: 200, height: 140 }, 100, -100, 0.5),
    { x: 0, y: -40, width: 200, height: 140 },
  );
  assert.deepEqual(
    moveRect(100, 100, { x: 25, y: 25, width: 50, height: 50 }, -100, 100, 1),
    { x: -25, y: 75, width: 50, height: 50 },
  );
});

test("corner resizing stays proportional and respects scale limits", () => {
  const resized = resizeRectFromDelta(
    { x: 25, y: 25, width: 50, height: 25 },
    "se",
    40,
    20,
    2,
    5,
    80,
  );
  close(resized.width / resized.height, 2);
  assert.equal(Math.max(resized.width, resized.height), 80);
  assert.equal(resized.x, 25);
  assert.equal(resized.y, 25);
});

test("placement JSON is normalized and deterministically ordered", () => {
  const parsed = parsePlacementData('{"version":1,"workspace_padding":2,"layer_order":["foreground_10","foreground_2","foreground_10"],"layers":{"foreground_10":{"scale":2},"foreground_2":{}}}');
  assert.deepEqual(parsed.layers.foreground_2, { scale: 0.9, long_axis_shift: 0, short_axis_shift: 0, flip_horizontal: false });
  assert.equal(parsed.workspace_padding, 1);
  assert.deepEqual(parsed.layer_order, ["foreground_10", "foreground_2"]);
  assert.equal(
    serializePlacementData(parsed),
    '{"version":1,"workspace_padding":1,"layer_order":["foreground_10","foreground_2"],"layers":{"foreground_2":{"scale":0.9,"long_axis_shift":0,"short_axis_shift":0,"flip_horizontal":false},"foreground_10":{"scale":2,"long_axis_shift":0,"short_axis_shift":0,"flip_horizontal":false}}}',
  );
  assert.deepEqual(parsePlacementData("not json"), { version: 2, workspace_padding: 0.5, layer_order: [], layers: {} });
});

test("version 2 stores normalized centers outside the output canvas", () => {
  const placement = rectToPlacement(100, 80, { x: -20, y: 60, width: 40, height: 20 });
  assert.deepEqual(placement, { scale: 0.5, center_x: 0, center_y: 0.875, flip_horizontal: false });
  assert.deepEqual(placementToRect(100, 80, 2, placement), { x: -20, y: 60, width: 40, height: 20 });
  const encoded = serializePlacementData({
    version: 2,
    workspace_padding: 1,
    layer_order: [],
    layers: { foreground_0: placement },
  });
  assert.equal(encoded, '{"version":2,"workspace_padding":1,"layer_order":[],"layers":{"foreground_0":{"scale":0.5,"center_x":0,"center_y":0.875,"flip_horizontal":false}}}');
});

test("horizontal flip survives placement normalization and geometry edits", () => {
  const parsed = parsePlacementData('{"version":2,"layers":{"foreground_0":{"flip_horizontal":true}}}');
  assert.equal(parsed.layers.foreground_0.flip_horizontal, true);
  const rect = placementToRect(100, 80, 2, parsed.layers.foreground_0);
  assert.equal(rectToPlacement(100, 80, rect, parsed.layers.foreground_0).flip_horizontal, true);
});
