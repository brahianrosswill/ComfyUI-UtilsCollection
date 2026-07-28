import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "ComfyUI.UtilsCollection.MarkdownPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "UC_MarkdownPreview") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const markdownWidget = ComfyWidgets.MARKDOWN(
        this,
        "markdown",
        ["STRING", { default: "" }],
        app,
      ).widget;
      markdownWidget.serialize = true;
      this.serialize_widgets = true;
      this.__ucMarkdownWidget = markdownWidget;
      this.setSize([
        Math.max(this.size[0], 360),
        Math.max(this.size[1], 240),
      ]);
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const markdown = message?.markdown;
      const value = Array.isArray(markdown) ? markdown[0] : markdown;
      if (this.__ucMarkdownWidget && value !== undefined) {
        this.__ucMarkdownWidget.value = String(value);
        this.setDirtyCanvas(true, true);
      }
    };
  },
});
