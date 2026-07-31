const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;

export const EMPTY_PREVIEW_ASPECT = 16 / 9;
export const VISIBLE_LAYER_ROWS = 3;
export const BACKGROUND_PREVIEW_ALPHA = 1;

export function flexContentWidth(itemWidths = [], gap = 0, chromeWidth = 0) {
  const widths = itemWidths.map(finite);
  const roundingAllowance = widths.length;
  return Math.ceil(
    widths.reduce((total, width) => total + width, 0)
    + Math.max(0, widths.length - 1) * finite(gap)
    + finite(chromeWidth)
    + roundingAllowance
  );
}

export function inlinePanelLayout(baseNodeWidth, chromeWidth, panelWidth = 0, gap = 0) {
  const extraWidth = Math.max(0, finite(panelWidth)) > 0
    ? Math.max(0, finite(panelWidth)) + Math.max(0, finite(gap))
    : 0;
  const nodeWidth = Math.max(0, finite(baseNodeWidth)) + extraWidth;
  return {
    extraWidth,
    nodeWidth,
    previewWidth: Math.max(0, nodeWidth - Math.max(0, finite(chromeWidth)) - extraWidth),
  };
}

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
