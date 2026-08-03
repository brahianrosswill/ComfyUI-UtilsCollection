import assert from "node:assert/strict";
import test from "node:test";

import {
  annotatedImagePath,
  applyImageWidgetValue,
  droppedImageFiles,
  retainPropertyAssignedImage,
} from "../web/load_image_alpha.js";

test("annotated image paths preserve upload location", () => {
  assert.equal(annotatedImagePath({ name: "a.png", subfolder: "", type: "input" }), "a.png");
  assert.equal(annotatedImagePath({ name: "a.png", subfolder: "clipspace", type: "input" }), "clipspace/a.png");
  assert.equal(annotatedImagePath({ name: "a.png", subfolder: "clipspace", type: "temp" }), "clipspace/a.png [temp]");
});

test("drop filtering accepts image MIME types and known image extensions", () => {
  const files = [
    { name: "alpha.png", type: "image/png" },
    { name: "alpha.webp", type: "" },
    { name: "notes.txt", type: "text/plain" },
  ];
  assert.deepEqual(droppedImageFiles({ dataTransfer: { files } }), files.slice(0, 2));
});

test("image values are retained in the combo and serialized widget slot", () => {
  const widget = { name: "image", value: "old.png", options: { values: ["old.png"] } };
  const node = { widgets: [widget], widgets_values: ["old.png"] };
  applyImageWidgetValue(node, widget, "clipspace/edited.png");
  assert.equal(widget.value, "clipspace/edited.png");
  assert.deepEqual(widget.options.values, ["old.png", "clipspace/edited.png"]);
  assert.equal(node.widgets_values[0], "clipspace/edited.png");
});

test("Mask Editor property assignment retains its generated image", () => {
  const widget = { name: "image", value: "old.png", options: { values: ["old.png"] } };
  const node = { properties: {}, widgets: [widget], widgets_values: ["old.png"] };
  retainPropertyAssignedImage(node, widget);
  node.properties.image = "clipspace/edited.png [temp]";
  assert.equal(widget.value, "clipspace/edited.png [temp]");
  assert.ok(widget.options.values.includes("clipspace/edited.png [temp]"));
});
