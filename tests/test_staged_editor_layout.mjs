import test from "node:test";
import assert from "node:assert/strict";

import {
  BACKGROUND_PREVIEW_ALPHA,
  editorWidgetHeight,
  floatingPanelPosition,
  growNodeSize,
  naturalEditorWidth,
  previewHeight,
  visibleLayerListHeight,
} from "../web/staged_editor_layout.js";

test("background preview is not arbitrarily dimmed", () => {
  assert.equal(BACKGROUND_PREVIEW_ALPHA, 1);
});

test("natural editor width follows its widest complete control row", () => {
  assert.equal(naturalEditorWidth({
    coreWidth: 300, toolbarWidth: 620, rowWidths: [480, 840], overlayWidths: [410], chromeWidth: 20,
  }), 860);
});

test("preview uses 16:9 without an image and the real image aspect when known", () => {
  assert.equal(previewHeight(800), 450);
  assert.equal(previewHeight(800, 1600, 800), 400);
  assert.equal(previewHeight(800, 800, 1600), 1600);
  assert.equal(previewHeight(800, 800, 800), 800);
});

test("layer list exposes three complete measured rows", () => {
  assert.equal(visibleLayerListHeight([40, 42, 44, 50], 3, 6), 138);
  assert.equal(visibleLayerListHeight([40, 42], 3, 6), 91);
});

test("widget height is composed from measured sections", () => {
  assert.equal(editorWidgetHeight({
    stageHeight: 450, toolbarHeight: 34, layerGroupHeight: 150, chromeHeight: 10, gaps: 12,
  }), 656);
});

test("automatic sizing grows but never shrinks a node", () => {
  assert.deepEqual(growNodeSize([900, 700], [800, 600]), [900, 700]);
  assert.deepEqual(growNodeSize([700, 500], [800, 600]), [800, 600]);
  assert.deepEqual(growNodeSize([800, 600], [800, 600]), [800, 600]);
});

test("floating picker prefers right, then left, then vertical placement", () => {
  const base = {
    avoid: { left: 300, right: 700, top: 100, bottom: 600 },
    trigger: { left: 320, top: 300, width: 30, height: 28 },
    panelWidth: 220, panelHeight: 330, viewportWidth: 1200, viewportHeight: 800,
  };
  assert.deepEqual(floatingPanelPosition(base), { left: 708, top: 149 });
  assert.deepEqual(floatingPanelPosition({ ...base, viewportWidth: 900 }), { left: 72, top: 149 });
  assert.deepEqual(floatingPanelPosition({
    ...base,
    avoid: { left: 120, right: 780, top: 200, bottom: 500 },
    trigger: { left: 400, top: 300, width: 30, height: 28 },
    viewportWidth: 900, viewportHeight: 900,
  }), { left: 305, top: 508 });
});
