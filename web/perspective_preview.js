import { projectQuadPoint } from "./placement_geometry.js";

export function drawProjectiveImage(context, image, rect, corners, flipHorizontal = false, flipVertical = false) {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  if (!sourceWidth || !sourceHeight) return;
  const meshSize = 8;
  const destination = (u, v) => {
    const [x, y] = projectQuadPoint(corners, u, v);
    return [rect.x + (x + 1) * rect.width / 2, rect.y + (y + 1) * rect.height / 2];
  };
  const source = (u, v) => [
    (flipHorizontal ? 1 - u : u) * sourceWidth,
    (flipVertical ? 1 - v : v) * sourceHeight,
  ];
  const drawTriangle = (sourcePoints, destinationPoints) => {
    const [[sx0, sy0], [sx1, sy1], [sx2, sy2]] = sourcePoints;
    const [[dx0, dy0], [dx1, dy1], [dx2, dy2]] = destinationPoints;
    const denominator = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);
    if (Math.abs(denominator) < 1e-8) return;
    const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denominator;
    const b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denominator;
    const c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denominator;
    const d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denominator;
    const e = (
      dx0 * (sx1 * sy2 - sx2 * sy1)
      + dx1 * (sx2 * sy0 - sx0 * sy2)
      + dx2 * (sx0 * sy1 - sx1 * sy0)
    ) / denominator;
    const f = (
      dy0 * (sx1 * sy2 - sx2 * sy1)
      + dy1 * (sx2 * sy0 - sx0 * sy2)
      + dy2 * (sx0 * sy1 - sx1 * sy0)
    ) / denominator;
    context.save();
    context.beginPath();
    context.moveTo(dx0, dy0);
    context.lineTo(dx1, dy1);
    context.lineTo(dx2, dy2);
    context.closePath();
    context.clip();
    context.transform(a, b, c, d, e, f);
    context.drawImage(image, 0, 0);
    context.restore();
  };
  for (let row = 0; row < meshSize; row++) {
    for (let column = 0; column < meshSize; column++) {
      const u0 = column / meshSize, u1 = (column + 1) / meshSize;
      const v0 = row / meshSize, v1 = (row + 1) / meshSize;
      const sourcePoints = [source(u0, v0), source(u1, v0), source(u1, v1), source(u0, v1)];
      const destinationPoints = [destination(u0, v0), destination(u1, v0), destination(u1, v1), destination(u0, v1)];
      drawTriangle(sourcePoints.slice(0, 3), destinationPoints.slice(0, 3));
      drawTriangle(
        [sourcePoints[0], sourcePoints[2], sourcePoints[3]],
        [destinationPoints[0], destinationPoints[2], destinationPoints[3]],
      );
    }
  }
}
