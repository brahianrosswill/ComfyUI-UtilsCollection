import test from "node:test";
import assert from "node:assert/strict";

import {
  BACKGROUND_PREVIEW_ALPHA,
  editorWidgetHeight,
  flexContentWidth,
  growNodeSize,
  inlinePanelLayout,
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

test("complete flex rows accumulate controls, gaps, chrome, and rounding allowance", () => {
  assert.equal(flexContentWidth([28, 24, 48, 32, 820], 3, 8), 977);
  assert.equal(naturalEditorWidth({
    coreWidth: 300, toolbarWidth: 620, rowWidths: [977], overlayWidths: [410], chromeWidth: 20,
  }), 997);
});

test("inline picker expands node without changing preview width", () => {
  assert.deepEqual(inlinePanelLayout(900, 16, 0, 0), {
    extraWidth: 0, nodeWidth: 900, previewWidth: 884,
  });
  assert.deepEqual(inlinePanelLayout(900, 16, 220, 6), {
    extraWidth: 226, nodeWidth: 1126, previewWidth: 884,
  });
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

test("required sizing grows from the manual size floor rather than previous automatic growth", () => {
  assert.deepEqual(growNodeSize([900, 700], [800, 600]), [900, 700]);
  assert.deepEqual(growNodeSize([700, 500], [800, 600]), [800, 600]);
  assert.deepEqual(growNodeSize([800, 600], [800, 600]), [800, 600]);
});
