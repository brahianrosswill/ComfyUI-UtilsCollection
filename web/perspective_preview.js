const interpolate = (a, b, amount) => [
  a[0] + (b[0] - a[0]) * amount,
  a[1] + (b[1] - a[1]) * amount,
];

function projectPoint(points, u, v) {
  const [topLeft, topRight, bottomRight, bottomLeft] = points;
  const dx1 = topRight[0] - bottomRight[0];
  const dx2 = bottomLeft[0] - bottomRight[0];
  const dx3 = topLeft[0] - topRight[0] + bottomRight[0] - bottomLeft[0];
  const dy1 = topRight[1] - bottomRight[1];
  const dy2 = bottomLeft[1] - bottomRight[1];
  const dy3 = topLeft[1] - topRight[1] + bottomRight[1] - bottomLeft[1];
  const denominator = dx1 * dy2 - dx2 * dy1;
  const g = Math.abs(denominator) < 1e-12 ? 0 : (dx3 * dy2 - dx2 * dy3) / denominator;
  const h = Math.abs(denominator) < 1e-12 ? 0 : (dx1 * dy3 - dx3 * dy1) / denominator;
  const a = topRight[0] - topLeft[0] + g * topRight[0];
  const b = bottomLeft[0] - topLeft[0] + h * bottomLeft[0];
  const d = topRight[1] - topLeft[1] + g * topRight[1];
  const e = bottomLeft[1] - topLeft[1] + h * bottomLeft[1];
  const scale = g * u + h * v + 1;
  return [
    (a * u + b * v + topLeft[0]) / scale,
    (d * u + e * v + topLeft[1]) / scale,
  ];
}

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function projectiveCellError(points, u0, v0, u1, v1) {
  const um = (u0 + u1) / 2, vm = (v0 + v1) / 2;
  const topLeft = projectPoint(points, u0, v0);
  const topRight = projectPoint(points, u1, v0);
  const bottomRight = projectPoint(points, u1, v1);
  const bottomLeft = projectPoint(points, u0, v1);
  const exact = [
    projectPoint(points, um, v0),
    projectPoint(points, u1, vm),
    projectPoint(points, um, v1),
    projectPoint(points, u0, vm),
    projectPoint(points, um, vm),
  ];
  const approximate = [
    interpolate(topLeft, topRight, 0.5),
    interpolate(topRight, bottomRight, 0.5),
    interpolate(bottomLeft, bottomRight, 0.5),
    interpolate(topLeft, bottomLeft, 0.5),
    interpolate(interpolate(topLeft, topRight, 0.5), interpolate(bottomLeft, bottomRight, 0.5), 0.5),
  ];
  return Math.max(...exact.map((point, index) => distance(point, approximate[index])));
}

export function drawProjectiveImage(
  context,
  image,
  destinationPoints,
  flipHorizontal = false,
  flipVertical = false,
  { errorTolerance = 0.5, maxDepth = 5 } = {},
) {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  if (!sourceWidth || !sourceHeight || destinationPoints?.length !== 4) return;
  const destination = (u, v) => projectPoint(destinationPoints, u, v);
  const source = (u, v) => [
    (flipHorizontal ? 1 - u : u) * sourceWidth,
    (flipVertical ? 1 - v : v) * sourceHeight,
  ];
  const drawTriangle = (sourcePoints, targetPoints) => {
    const [[sx0, sy0], [sx1, sy1], [sx2, sy2]] = sourcePoints;
    const [[dx0, dy0], [dx1, dy1], [dx2, dy2]] = targetPoints;
    const denominator = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);
    if (Math.abs(denominator) < 1e-8) return;
    const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denominator;
    const b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denominator;
    const c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denominator;
    const d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denominator;
    const e = (dx0 * (sx1 * sy2 - sx2 * sy1) + dx1 * (sx2 * sy0 - sx0 * sy2) + dx2 * (sx0 * sy1 - sx1 * sy0)) / denominator;
    const f = (dy0 * (sx1 * sy2 - sx2 * sy1) + dy1 * (sx2 * sy0 - sx0 * sy2) + dy2 * (sx0 * sy1 - sx1 * sy0)) / denominator;
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
  const drawCell = (u0, v0, u1, v1, depth) => {
    if (depth < maxDepth && projectiveCellError(destinationPoints, u0, v0, u1, v1) > errorTolerance) {
      const um = (u0 + u1) / 2, vm = (v0 + v1) / 2;
      drawCell(u0, v0, um, vm, depth + 1);
      drawCell(um, v0, u1, vm, depth + 1);
      drawCell(um, vm, u1, v1, depth + 1);
      drawCell(u0, vm, um, v1, depth + 1);
      return;
    }
    const sourcePoints = [source(u0, v0), source(u1, v0), source(u1, v1), source(u0, v1)];
    const targetPoints = [destination(u0, v0), destination(u1, v0), destination(u1, v1), destination(u0, v1)];
    drawTriangle(sourcePoints.slice(0, 3), targetPoints.slice(0, 3));
    drawTriangle([sourcePoints[0], sourcePoints[2], sourcePoints[3]], [targetPoints[0], targetPoints[2], targetPoints[3]]);
  };
  drawCell(0, 0, 1, 1, 0);
}
