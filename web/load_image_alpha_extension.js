import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  droppedImageFiles,
  retainPropertyAssignedImage,
  uploadDroppedImages,
} from "./load_image_alpha.js";

app.registerExtension({
  name: "ComfyUI.UtilsCollection.LoadImageAlphaCompatibility",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "UC_LoadImageWithAlpha") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const imageWidget = this.widgets?.find((widget) => widget.name === "image");
      if (!imageWidget) return result;

      retainPropertyAssignedImage(this, imageWidget);
      this.onDragOver = (event) => {
        const items = Array.from(event?.dataTransfer?.items || []);
        return items.some((item) => item.kind === "file");
      };
      this.onDragDrop = async (event) => {
        const files = droppedImageFiles(event);
        if (!files.length) return false;
        await uploadDroppedImages(api, this, imageWidget, files);
        return true;
      };
      return result;
    };
  },
});
