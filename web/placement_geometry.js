export const DEFAULT_PLACEMENT = Object.freeze({
  scale: 0.9,
  long_axis_shift: 0,
  short_axis_shift: 0,
});
export const DEFAULT_WORKSPACE_PADDING = 0.5;

const finite = (value, fallback) => Number.isFinite(Number(value)) ? Number(value) : fallback;
export const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
export const normalizeWorkspacePadding = (value) => clamp(finite(value, DEFAULT_WORKSPACE_PADDING), 0, 1);

export function normalizePlacement(value = {}) {
  return {
    scale: clamp(finite(value.scale, DEFAULT_PLACEMENT.scale), 0.05, 10),
    long_axis_shift: clamp(finite(value.long_axis_shift, 0), -1, 1),
    short_axis_shift: clamp(finite(value.short_axis_shift, 0), -1, 1),
  };
}

export function parsePlacementData(value) {
  try {
    const data = typeof value === "string" ? JSON.parse(value || "{}") : value;
    if (!data || typeof data !== "object" || (data.version ?? 1) !== 1) throw new Error();
    const layers = {};
    for (const [key, placement] of Object.entries(data.layers || {})) {
      if (placement && typeof placement === "object") layers[key] = normalizePlacement(placement);
    }
    return { version: 1, workspace_padding: normalizeWorkspacePadding(data.workspace_padding), layers };
  } catch {
    return { version: 1, workspace_padding: DEFAULT_WORKSPACE_PADDING, layers: {} };
  }
}

export function serializePlacementData(data) {
  const layers = {};
  for (const key of Object.keys(data.layers || {}).sort(layerKeyCompare)) {
    layers[key] = normalizePlacement(data.layers[key]);
  }
  return JSON.stringify({
    version: 1,
    workspace_padding: normalizeWorkspacePadding(data.workspace_padding),
    layers,
  });
}

export function layerKeyCompare(a, b) {
  const ai = Number((a.match(/\d+/) || [0])[0]);
  const bi = Number((b.match(/\d+/) || [0])[0]);
  return ai - bi || a.localeCompare(b);
}

export function sizeFromScale(backgroundWidth, backgroundHeight, aspect, scale) {
  const longest = Math.max(0, Number(scale)) * Math.min(backgroundWidth, backgroundHeight);
  const ratio = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  return ratio >= 1
    ? { width: longest, height: longest / ratio }
    : { width: longest * ratio, height: longest };
}

function physicalShifts(backgroundWidth, backgroundHeight, placement) {
  if (backgroundWidth >= backgroundHeight) {
    return { x: placement.long_axis_shift, y: placement.short_axis_shift };
  }
  return { x: placement.short_axis_shift, y: placement.long_axis_shift };
}

function logicalShifts(backgroundWidth, backgroundHeight, x, y) {
  return backgroundWidth >= backgroundHeight
    ? { long_axis_shift: x, short_axis_shift: y }
    : { long_axis_shift: y, short_axis_shift: x };
}

export function placementToRect(backgroundWidth, backgroundHeight, aspect, value) {
  const placement = normalizePlacement(value);
  const size = sizeFromScale(backgroundWidth, backgroundHeight, aspect, placement.scale);
  const shifts = physicalShifts(backgroundWidth, backgroundHeight, placement);
  const travelX = backgroundWidth - size.width;
  const travelY = backgroundHeight - size.height;
  return {
    x: ((shifts.x + 1) / 2) * travelX,
    y: ((shifts.y + 1) / 2) * travelY,
    width: size.width,
    height: size.height,
  };
}

function shiftFromOffset(offset, travel, prior) {
  if (Math.abs(travel) < 1e-9) return clamp(finite(prior, 0), -1, 1);
  return clamp((offset / travel) * 2 - 1, -1, 1);
}

export function rectToPlacement(backgroundWidth, backgroundHeight, rect, prior = DEFAULT_PLACEMENT) {
  const longest = Math.max(rect.width, rect.height);
  const scale = longest / Math.min(backgroundWidth, backgroundHeight);
  const priorPhysical = physicalShifts(backgroundWidth, backgroundHeight, normalizePlacement(prior));
  const x = shiftFromOffset(rect.x, backgroundWidth - rect.width, priorPhysical.x);
  const y = shiftFromOffset(rect.y, backgroundHeight - rect.height, priorPhysical.y);
  return normalizePlacement({ scale, ...logicalShifts(backgroundWidth, backgroundHeight, x, y) });
}

export function moveRect(backgroundWidth, backgroundHeight, rect, deltaX, deltaY) {
  const travelX = backgroundWidth - rect.width;
  const travelY = backgroundHeight - rect.height;
  return {
    ...rect,
    x: clamp(rect.x + deltaX, Math.min(0, travelX), Math.max(0, travelX)),
    y: clamp(rect.y + deltaY, Math.min(0, travelY), Math.max(0, travelY)),
  };
}

export function drawRect(start, end, aspect) {
  const ratio = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const availableWidth = Math.abs(end.x - start.x);
  const availableHeight = Math.abs(end.y - start.y);
  const width = Math.min(availableWidth, availableHeight * ratio);
  const height = width / ratio;
  return {
    x: Math.min(start.x, end.x) + (availableWidth - width) / 2,
    y: Math.min(start.y, end.y) + (availableHeight - height) / 2,
    width,
    height,
  };
}

export function resizeRectFromDelta(rect, handle, deltaX, deltaY, aspect, minimumLongest, maximumLongest) {
  const ratio = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const widthPerLongest = ratio >= 1 ? 1 : ratio;
  const heightPerLongest = ratio >= 1 ? 1 / ratio : 1;
  const signX = handle.includes("w") ? -1 : 1;
  const signY = handle.includes("n") ? -1 : 1;
  const projected = (
    deltaX * signX * widthPerLongest + deltaY * signY * heightPerLongest
  ) / (widthPerLongest ** 2 + heightPerLongest ** 2);
  const startLongest = Math.max(rect.width, rect.height);
  const longest = clamp(startLongest + projected, minimumLongest, maximumLongest);
  const width = longest * widthPerLongest;
  const height = longest * heightPerLongest;
  const fixedX = handle.includes("w") ? rect.x + rect.width : rect.x;
  const fixedY = handle.includes("n") ? rect.y + rect.height : rect.y;
  return {
    x: handle.includes("w") ? fixedX - width : fixedX,
    y: handle.includes("n") ? fixedY - height : fixedY,
    width,
    height,
  };
}
