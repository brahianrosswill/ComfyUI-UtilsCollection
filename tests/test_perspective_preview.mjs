import assert from "node:assert/strict";
import test from "node:test";

import { drawProjectiveImage, projectiveCellError } from "../web/perspective_preview.js";

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
    [[10, 20], [180, 36], [190, 148], [20, 180]],
  );
  assert.ok(draws >= 2 && draws <= 2048);
  assert.equal(transforms.length, draws);
  assert.ok(transforms.flat().every(Number.isFinite));
});

test("adaptive projective preview subdivides only beyond its error tolerance", () => {
  const affine = [[0, 0], [100, 0], [100, 100], [0, 100]];
  const perspective = [[0, 0], [100, 10], [80, 100], [20, 80]];
  assert.ok(projectiveCellError(affine, 0, 0, 1, 1) < 1e-9);
  assert.ok(projectiveCellError(perspective, 0, 0, 1, 1) > 0.5);
});
