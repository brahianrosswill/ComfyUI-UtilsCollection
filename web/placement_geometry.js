export const DEFAULT_PLACEMENT = Object.freeze({
  scale: 0.9,
  center_x: 0.5,
  center_y: 0.5,
  flip_horizontal: false,
});
export const LEGACY_DEFAULT_PLACEMENT = Object.freeze({ scale: 0.9, long_axis_shift: 0, short_axis_shift: 0 });
export const DEFAULT_WORKSPACE_PADDING = 0.5;

const finite = (value, fallback) => Number.isFinite(Number(value)) ? Number(value) : fallback;
export const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
export const normalizeWorkspacePadding = (value) => clamp(finite(value, DEFAULT_WORKSPACE_PADDING), 0, 1);

export function normalizePlacement(value = {}, version = 2) {
  const scale = clamp(finite(value.scale, DEFAULT_PLACEMENT.scale), 0.05, 10);
  if (version === 1 && value.center_x === undefined && value.center_y === undefined) {
    return {
      scale,
      long_axis_shift: clamp(finite(value.long_axis_shift, 0), -1, 1),
      short_axis_shift: clamp(finite(value.short_axis_shift, 0), -1, 1),
      flip_horizontal: value.flip_horizontal === true,
    };
  }
  return {
    scale,
    center_x: clamp(finite(value.center_x, 0.5), -10, 10),
    center_y: clamp(finite(value.center_y, 0.5), -10, 10),
    flip_horizontal: value.flip_horizontal === true,
  };
}

export function parsePlacementData(value) {
  try {
    const data = typeof value === "string" ? JSON.parse(value || "{}") : value;
    const version = data?.version ?? 1;
    if (!data || typeof data !== "object" || ![1, 2].includes(version)) throw new Error();
    const layers = {};
    for (const [key, placement] of Object.entries(data.layers || {})) {
      if (placement && typeof placement === "object") layers[key] = normalizePlacement(placement, version);
    }
    const layer_order = Array.isArray(data.layer_order)
      ? [...new Set(data.layer_order.filter((key) => typeof key === "string"))]
      : [];
    return { version, workspace_padding: normalizeWorkspacePadding(data.workspace_padding), layer_order, layers };
  } catch {
    return { version: 2, workspace_padding: DEFAULT_WORKSPACE_PADDING, layer_order: [], layers: {} };
  }
}

export function serializePlacementData(data) {
  const layers = {};
  for (const key of Object.keys(data.layers || {}).sort(layerKeyCompare)) {
    layers[key] = normalizePlacement(data.layers[key], data.version ?? 2);
  }
  return JSON.stringify({
    version: data.version ?? 2,
    workspace_padding: normalizeWorkspacePadding(data.workspace_padding),
    layer_order: Array.isArray(data.layer_order)
      ? [...new Set(data.layer_order.filter((key) => typeof key === "string"))]
      : [],
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

export function placementToRect(backgroundWidth, backgroundHeight, aspect, value) {
  const legacy = value?.center_x === undefined && value?.center_y === undefined;
  const placement = normalizePlacement(value, legacy ? 1 : 2);
  const size = sizeFromScale(backgroundWidth, backgroundHeight, aspect, placement.scale);
  if (!legacy) {
    return {
      x: placement.center_x * backgroundWidth - size.width / 2,
      y: placement.center_y * backgroundHeight - size.height / 2,
      width: size.width,
      height: size.height,
    };
  }
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

export function rectToPlacement(backgroundWidth, backgroundHeight, rect, prior = {}) {
  const longest = Math.max(rect.width, rect.height);
  const scale = longest / Math.min(backgroundWidth, backgroundHeight);
  return normalizePlacement({
    scale,
    center_x: (rect.x + rect.width / 2) / backgroundWidth,
    center_y: (rect.y + rect.height / 2) / backgroundHeight,
    flip_horizontal: prior.flip_horizontal === true,
  });
}

export function moveRect(backgroundWidth, backgroundHeight, rect, deltaX, deltaY, workspacePadding = 0) {
  const padding = normalizeWorkspacePadding(workspacePadding);
  const paddingX = backgroundWidth * 0.25 * padding;
  const paddingY = backgroundHeight * 0.25 * padding;
  const travelX = backgroundWidth + paddingX * 2 - rect.width;
  const travelY = backgroundHeight + paddingY * 2 - rect.height;
  const xLimits = [-paddingX, -paddingX + travelX, 0, backgroundWidth - rect.width];
  const yLimits = [-paddingY, -paddingY + travelY, 0, backgroundHeight - rect.height];
  return {
    ...rect,
    x: clamp(rect.x + deltaX, Math.min(...xLimits), Math.max(...xLimits)) || 0,
    y: clamp(rect.y + deltaY, Math.min(...yLimits), Math.max(...yLimits)) || 0,
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
