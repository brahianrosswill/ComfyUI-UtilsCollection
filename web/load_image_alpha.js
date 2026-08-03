export function isImageFile(file) {
  if (!file) return false;
  if (String(file.type || "").startsWith("image/")) return true;
  return /\.(?:avif|bmp|gif|jpe?g|png|tiff?|webp)$/i.test(String(file.name || ""));
}

export function droppedImageFiles(event) {
  return Array.from(event?.dataTransfer?.files || []).filter(isImageFile);
}

export function annotatedImagePath(uploaded) {
  const name = String(uploaded?.name || "");
  const subfolder = String(uploaded?.subfolder || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
  const path = subfolder ? `${subfolder}/${name}` : name;
  const type = String(uploaded?.type || "input");
  return type && type !== "input" ? `${path} [${type}]` : path;
}

export function retainComboValue(widget, value) {
  const values = widget?.options?.values;
  if (Array.isArray(values) && !values.includes(value)) values.push(value);
}

export function applyImageWidgetValue(node, widget, value, invokeCallback = true) {
  if (!widget || typeof value !== "string" || !value) return;
  retainComboValue(widget, value);
  const oldValue = widget.value;
  widget.value = value;
  const widgetIndex = node.widgets?.indexOf(widget) ?? -1;
  if (widgetIndex >= 0 && Array.isArray(node.widgets_values)) {
    node.widgets_values[widgetIndex] = value;
  }
  if (invokeCallback && oldValue !== value) {
    widget.callback?.(value);
    node.onWidgetChanged?.(widget.name, value, oldValue, widget);
  }
  node.imgs = undefined;
  node.graph?.setDirtyCanvas?.(true, true);
}

export function retainPropertyAssignedImage(node, widget) {
  const properties = node.properties;
  if (!properties || properties.__ucAlphaImagePropertyHook) return;
  let current = properties.image;
  Object.defineProperty(properties, "image", {
    configurable: true,
    enumerable: true,
    get: () => current,
    set: (value) => {
      current = value;
      applyImageWidgetValue(node, widget, value);
    },
  });
  Object.defineProperty(properties, "__ucAlphaImagePropertyHook", {
    configurable: true,
    value: true,
  });
}

export async function uploadDroppedImages(api, node, widget, files) {
  for (const file of files) {
    const form = new FormData();
    form.append("image", file);
    form.append("type", "input");
    const response = await api.fetchApi("/upload/image", {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Image upload failed (${response.status}${detail ? `: ${detail}` : ""}).`);
    }
    const uploaded = await response.json();
    const value = annotatedImagePath({ ...uploaded, type: "input" });
    applyImageWidgetValue(node, widget, value);
    if (node.properties) node.properties.image = value;
  }
}
