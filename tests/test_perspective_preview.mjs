import assert from "node:assert/strict";
import test from "node:test";

import { drawProjectiveImage } from "../web/perspective_preview.js";

test("projective preview draws a finite clipped mesh", () => {
  const transforms = [];
  let draws = 0;
  const context = {
    save() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    closePath() {},
    clip() {},
    transform(...values) { transforms.push(values); },
    drawImage() { draws += 1; },
    restore() {},
  };
  drawProjectiveImage(
    context,
    { naturalWidth: 100, naturalHeight: 80 },
    { x: 10, y: 20, width: 200, height: 160 },
    [[-1, -1], [0.7, -0.8], [0.8, 0.6], [-0.9, 1]],
  );
  assert.equal(draws, 128);
  assert.equal(transforms.length, draws);
  assert.ok(transforms.flat().every(Number.isFinite));
});
