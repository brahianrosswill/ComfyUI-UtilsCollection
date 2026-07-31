import test from "node:test";
import assert from "node:assert/strict";

import {
  brushTextureKey,
  interpolatedPoints,
  imageDataAlphaBounds,
  normalizeBrushSettings,
} from "../web/paint_brush.js";

test("brush settings clamp without losing supported shapes", () => {
  assert.deepEqual(normalizeBrushSettings({
    shape: "square", color: "#ABCDEF", size: 999, opacity: -1, hardness: 2, erasing: true,
  }), {
    shape: "square", color: "#abcdef", size: 250, opacity: 0, hardness: 1, erasing: true,
  });
});

test("missing opacity defaults to solid without replacing a saved value", () => {
  assert.equal(normalizeBrushSettings({}).opacity, 1);
  assert.equal(normalizeBrushSettings({ opacity: 0.7 }).opacity, 0.7);
});

test("alpha bounds isolate the retained undo patch for clear", () => {
  const data = new Uint8ClampedArray(5 * 4 * 4);
  data[(1 * 5 + 2) * 4 + 3] = 255;
  data[(3 * 5 + 4) * 4 + 3] = 1;
  assert.deepEqual(imageDataAlphaBounds({ data, width: 5, height: 4 }), {
    left: 2, top: 1, right: 5, bottom: 4,
  });
  assert.equal(imageDataAlphaBounds({ data: new Uint8ClampedArray(16), width: 2, height: 2 }), null);
});

test("brush cache identity includes opacity and every rendered control", () => {
  const base = { shape: "circle", color: "#123456", size: 10, opacity: 0.7, hardness: 0.5 };
  for (const changed of [
    { shape: "square" }, { color: "#654321" }, { size: 11 }, { opacity: 0.6 }, { hardness: 0.4 },
  ]) assert.notEqual(brushTextureKey(base), brushTextureKey({ ...base, ...changed }));
});

test("fast strokes receive stamps no farther apart than requested spacing", () => {
  const points = interpolatedPoints({ x: 0, y: 0 }, { x: 19, y: 0 }, 2);
  assert.deepEqual(points.at(-1), { x: 19, y: 0 });
  let prior = { x: 0, y: 0 };
  for (const point of points) {
    assert.ok(Math.hypot(point.x - prior.x, point.y - prior.y) <= 2);
    prior = point;
  }
});
