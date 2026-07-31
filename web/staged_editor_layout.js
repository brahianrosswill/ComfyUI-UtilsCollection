const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;

export const EMPTY_PREVIEW_ASPECT = 16 / 9;
export const VISIBLE_LAYER_ROWS = 3;
export const BACKGROUND_PREVIEW_ALPHA = 1;

export function naturalEditorWidth({ coreWidth = 0, toolbarWidth = 0, rowWidths = [], overlayWidths = [], chromeWidth = 0 } = {}) {
  return Math.ceil(Math.max(
    finite(coreWidth),
    finite(toolbarWidth) + finite(chromeWidth),
    ...rowWidths.map((value) => finite(value) + finite(chromeWidth)),
    ...overlayWidths.map((value) => finite(value) + finite(chromeWidth)),
  ));
}

export function previewHeight(width, imageWidth = 0, imageHeight = 0) {
  const aspect = finite(imageWidth) > 0 && finite(imageHeight) > 0
    ? finite(imageWidth) / finite(imageHeight)
    : EMPTY_PREVIEW_ASPECT;
  return Math.ceil(Math.max(0, finite(width)) / aspect);
}

export function visibleLayerListHeight(rowHeights = [], gap = 0, chromeHeight = 0) {
  const visible = rowHeights.slice(0, VISIBLE_LAYER_ROWS).map(finite);
  return Math.ceil(
    visible.reduce((total, value) => total + value, 0)
    + Math.max(0, visible.length - 1) * finite(gap)
    + finite(chromeHeight)
  );
}

export function editorWidgetHeight({ stageHeight = 0, toolbarHeight = 0, layerGroupHeight = 0, chromeHeight = 0, gaps = 0 } = {}) {
  return Math.ceil(
    finite(stageHeight) + finite(toolbarHeight) + finite(layerGroupHeight)
    + finite(chromeHeight) + finite(gaps)
  );
}

export function growNodeSize(current = [0, 0], required = [0, 0]) {
  return [
    Math.max(finite(current[0]), finite(required[0])),
    Math.max(finite(current[1]), finite(required[1])),
  ];
}

export function floatingPanelPosition({ avoid, trigger, panelWidth, panelHeight, viewportWidth, viewportHeight, gap = 8, margin = 8 }) {
  const width = finite(panelWidth), height = finite(panelHeight);
  const viewport = { left: margin, top: margin, right: finite(viewportWidth) - margin, bottom: finite(viewportHeight) - margin };
  const fits = ({ left, top }) => left >= viewport.left && top >= viewport.top
    && left + width <= viewport.right && top + height <= viewport.bottom;
  const centeredTop = finite(trigger.top) + finite(trigger.height) / 2 - height / 2;
  const centeredLeft = finite(trigger.left) + finite(trigger.width) / 2 - width / 2;
  const candidates = [
    { left: finite(avoid.right) + gap, top: centeredTop },
    { left: finite(avoid.left) - gap - width, top: centeredTop },
    { left: centeredLeft, top: finite(avoid.bottom) + gap },
    { left: centeredLeft, top: finite(avoid.top) - gap - height },
  ];
  const selected = candidates.find(fits);
  if (selected) return { left: Math.round(selected.left), top: Math.round(selected.top) };
  const clamp = (value, minimum, maximum) => Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
  return {
    left: Math.round(clamp(centeredLeft, viewport.left, viewport.right - width)),
    top: Math.round(clamp(centeredTop, viewport.top, viewport.bottom - height)),
  };
}
