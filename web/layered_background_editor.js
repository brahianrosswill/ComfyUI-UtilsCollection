import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  DEFAULT_PLACEMENT,
  drawRect,
  layerKeyCompare,
  moveRect,
  normalizePlacement,
  parsePlacementData,
  placementToRect,
  rectToPlacement,
  resizeRectFromDelta,
  serializePlacementData,
} from "./placement_geometry.js";

const NODE_TYPES = new Set([
  "UC_LayeredBackgroundComposite",
  "UC_StagedLayeredBackgroundComposite",
]);
const editors = new Set();
let latestOutputs = {};

function descriptorUrl(descriptor) {
  if (!descriptor?.filename) return null;
  const query = new URLSearchParams({
    filename: descriptor.filename,
    subfolder: descriptor.subfolder || "",
    type: descriptor.type || "temp",
  });
  return api.apiURL(`/view?${query}`);
}

function style(element, values) {
  Object.assign(element.style, values);
  return element;
}

function element(tag, values = {}) {
  return style(document.createElement(tag), values);
}

class LayeredPlacementEditor {
  constructor(node) {
    this.node = node;
    this.placementWidget = node.widgets?.find((widget) => widget.name === "placement_data");
    if (!this.placementWidget) throw new Error("Layered compositor placement_data widget was not created.");
    this.data = parsePlacementData(this.placementWidget.value);
    this.selected = null;
    this.backgroundImage = null;
    this.backgroundSource = null;
    this.metadata = null;
    this.metadataSignature = null;
    this.cutouts = new Map();
    this.backgroundGeneration = 0;
    this.metadataGeneration = 0;
    this.drawFrame = 0;
    this.commitFrame = 0;
    this.pendingPlacement = null;
    this.gesture = null;
    this.keyTransaction = false;
    this.disposed = false;
    this.createDom();
    this.installWidget();
    this.wrapNodeLifecycle();
    this.refreshSources();
    this.pollTimer = window.setInterval(() => this.refreshSources(), 500);
  }

  createDom() {
    this.root = element("div", {
      boxSizing: "border-box",
      display: "flex",
      flexDirection: "column",
      gap: "6px",
      width: "100%",
      padding: "4px",
      color: "var(--input-text, #ddd)",
      font: "12px sans-serif",
    });
    this.stage = element("div", {
      position: "relative",
      minHeight: "260px",
      flex: "1 1 auto",
      overflow: "hidden",
      border: "1px solid rgba(255,255,255,.18)",
      borderRadius: "5px",
      background: "#17191d",
    });
    this.canvas = element("canvas", {
      display: "block",
      width: "100%",
      height: "100%",
      outline: "none",
      touchAction: "none",
    });
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("aria-label", "Layered foreground placement canvas");
    this.status = element("div", {
      position: "absolute",
      left: "8px",
      bottom: "7px",
      padding: "3px 6px",
      borderRadius: "4px",
      color: "#fff",
      background: "rgba(0,0,0,.65)",
      pointerEvents: "none",
    });
    this.stage.append(this.canvas, this.status);

    const controls = element("div", {
      display: "grid",
      gridTemplateColumns: "minmax(110px, 1fr) repeat(3, minmax(58px, .6fr)) auto",
      gap: "5px",
      alignItems: "end",
    });
    const layerGroup = this.labeledControl("Layer (back → front)");
    this.layerSelect = element("select", this.controlStyle());
    layerGroup.append(this.layerSelect);
    controls.append(layerGroup);
    this.inputs = {};
    for (const [field, label, step] of [
      ["scale", "Scale", "0.01"],
      ["long_axis_shift", "Long", "0.01"],
      ["short_axis_shift", "Short", "0.01"],
    ]) {
      const group = this.labeledControl(label);
      const input = element("input", this.controlStyle());
      input.type = "number";
      input.step = step;
      input.min = field === "scale" ? "0.05" : "-1";
      input.max = field === "scale" ? "10" : "1";
      input.dataset.field = field;
      input.addEventListener("change", () => this.numericChanged(field, input.value));
      input.addEventListener("keydown", (event) => event.stopPropagation());
      this.inputs[field] = input;
      group.append(input);
      controls.append(group);
    }
    this.resetButton = element("button", {
      ...this.controlStyle(),
      height: "24px",
      padding: "2px 9px",
      cursor: "pointer",
    });
    this.resetButton.type = "button";
    this.resetButton.textContent = "Reset";
    this.resetButton.title = "Reset the selected foreground placement";
    controls.append(this.resetButton);
    this.root.append(this.stage, controls);

    this.layerSelect.addEventListener("change", () => this.selectLayer(this.layerSelect.value));
    this.resetButton.addEventListener("click", () => this.resetSelected());
    for (const eventName of ["pointerdown", "pointermove", "pointerup", "wheel", "click", "dblclick"]) {
      this.root.addEventListener(eventName, (event) => event.stopPropagation());
    }
    this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    this.canvas.addEventListener("pointerup", (event) => this.pointerEnd(event, false));
    this.canvas.addEventListener("pointercancel", (event) => this.pointerEnd(event, true));
    this.canvas.addEventListener("keydown", (event) => this.keyDown(event));
    this.canvas.addEventListener("keyup", () => this.finishKeyTransaction());
    this.canvas.addEventListener("blur", () => this.finishKeyTransaction());
    this.resizeObserver = new ResizeObserver(() => this.requestDraw());
    this.resizeObserver.observe(this.stage);
  }

  labeledControl(label) {
    const group = element("label", { display: "flex", flexDirection: "column", gap: "2px", minWidth: "0" });
    const caption = document.createElement("span");
    caption.textContent = label;
    caption.style.opacity = ".78";
    group.append(caption);
    return group;
  }

  controlStyle() {
    return {
      boxSizing: "border-box",
      width: "100%",
      height: "23px",
      minWidth: "0",
      border: "1px solid rgba(255,255,255,.2)",
      borderRadius: "4px",
      color: "inherit",
      background: "rgba(0,0,0,.25)",
    };
  }

  installWidget() {
    this.placementWidget.computeSize = () => [0, -4];
    this.placementWidget.draw = () => {};
    const widget = this.node.addDOMWidget("layered_scene_editor", "uc_layered_scene_editor", this.root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => 370,
    });
    widget.serialize = false;
    const placementIndex = this.node.widgets.indexOf(this.placementWidget);
    const editorIndex = this.node.widgets.indexOf(widget);
    if (placementIndex >= 0 && editorIndex >= 0) {
      this.node.widgets.splice(editorIndex, 1);
      this.node.widgets.splice(placementIndex, 0, widget);
    }
    this.ensureSize();
  }

  ensureSize() {
    requestAnimationFrame(() => {
      if (this.disposed) return;
      const computed = this.node.computeSize?.() || this.node.size;
      this.node.setSize?.([
        Math.max(440, this.node.size?.[0] || 0),
        Math.max(computed?.[1] || 0, this.node.size?.[1] || 0),
      ]);
      this.node.graph?.setDirtyCanvas?.(true, true);
    });
  }

  wrapNodeLifecycle() {
    const originalConnections = this.node.onConnectionsChange;
    this.node.onConnectionsChange = (...args) => {
      const result = originalConnections?.apply(this.node, args);
      queueMicrotask(() => this.refreshSources(true));
      return result;
    };
    const originalRemoved = this.node.onRemoved;
    this.node.onRemoved = (...args) => {
      this.dispose();
      return originalRemoved?.apply(this.node, args);
    };
  }

  connectedLayers() {
    const direct = (this.node.inputs || [])
      .filter((input) => /foreground_\d+$/.test(input.name) && input.link != null)
      .map((input) => input.name.match(/foreground_\d+$/)[0])
      .sort(layerKeyCompare);
    if (direct.length) return direct;
    return (this.metadata?.layers || []).map((layer) => layer.socket).sort(layerKeyCompare);
  }

  inputOrigin(name) {
    const slot = (this.node.inputs || []).findIndex((input) => (
      input.name === name || input.name.endsWith(`.${name}`) || input.label === name
    ));
    const linkId = slot >= 0 ? this.node.inputs?.[slot]?.link : null;
    const link = linkId != null ? this.node.graph?.links?.[linkId] : null;
    return link ? { link, node: this.node.graph?._nodes_by_id?.[link.origin_id] } : null;
  }

  semanticSignature() {
    const sourceNames = (this.node.comfyClass === "UC_StagedLayeredBackgroundComposite" || this.node.type === "UC_StagedLayeredBackgroundComposite")
      ? ["background", "staged_foregrounds"]
      : ["background", ...this.connectedLayers()];
    const links = sourceNames.map((name) => {
      const origin = this.inputOrigin(name);
      const preview = origin?.node?.imgs?.[0];
      const previewIdentity = preview?.currentSrc || preview?.src || "";
      return `${name}:${origin?.link?.id ?? origin?.link?.origin_id ?? "-"}:${origin?.node?.mode ?? "-"}:${previewIdentity}`;
    });
    const maskValues = ["mask_threshold", "border_cleanup_width", "artifact_cleanup_radius", "gap_fill_radius", "feather_radius"]
      .map((name) => `${name}:${this.node.widgets?.find((widget) => widget.name === name)?.value}`);
    return [...links, ...maskValues].join("|");
  }

  refreshSources(force = false) {
    if (this.disposed) return;
    const layers = this.connectedLayers();
    if (!this.selected || !layers.includes(this.selected)) this.selected = layers[0] || null;
    this.syncLayerSelector(layers);
    const signature = this.semanticSignature();
    if (this.metadataSignature && signature !== this.metadataSignature) {
      this.metadata = null;
      this.metadataSignature = null;
      this.cutouts.clear();
      this.metadataGeneration++;
    }
    this.resolveBackground(force);
    this.syncNumericControls();
    this.requestDraw();
  }

  syncLayerSelector(layers) {
    const current = [...this.layerSelect.options].map((option) => option.value).join("|");
    if (current !== layers.join("|")) {
      this.layerSelect.replaceChildren(...layers.map((key, index) => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = `${key} (${index === 0 ? "back" : index === layers.length - 1 ? "front" : index + 1})`;
        return option;
      }));
    }
    this.layerSelect.value = this.selected || "";
    this.layerSelect.disabled = !this.selected;
  }

  resolveBackground(force) {
    const origin = this.inputOrigin("background");
    let source = null;
    let image = null;
    if (origin?.node && ![2, 4].includes(origin.node.mode)) {
      const candidate = origin.node.imgs?.[0];
      if (candidate?.complete && (candidate.naturalWidth || candidate.width)) {
        source = `element:${origin.link.origin_id}:${candidate.currentSrc || candidate.src || candidate.width}`;
        image = candidate;
      } else {
        const descriptor = latestOutputs[String(origin.link.origin_id)]?.images?.[0];
        const url = descriptorUrl(descriptor);
        if (url) source = url;
      }
    }
    if (!source && this.metadata && this.metadataSignature === this.semanticSignature()) {
      source = descriptorUrl(this.metadata.background?.preview);
    }
    if (!force && source === this.backgroundSource) return;
    this.backgroundSource = source;
    if (image) {
      this.backgroundGeneration++;
      this.backgroundImage = image;
      this.requestDraw();
    } else if (source) {
      const generation = ++this.backgroundGeneration;
      this.loadImage(source, (loaded) => {
        this.backgroundImage = loaded;
        this.requestDraw();
      }, () => generation === this.backgroundGeneration);
    } else {
      this.backgroundGeneration++;
      this.backgroundImage = null;
    }
  }

  loadImage(url, callback, isCurrent) {
    const image = new Image();
    image.onload = () => {
      if (!this.disposed && isCurrent()) callback(image);
    };
    image.onerror = () => this.requestDraw();
    image.src = url;
  }

  setOutput(output) {
    const metadata = output?.uc_layered_scene_editor?.[0];
    if (!metadata || metadata.version !== 1) return;
    this.metadata = metadata;
    this.metadataSignature = this.semanticSignature();
    this.cutouts.clear();
    const generation = ++this.metadataGeneration;
    for (const layer of metadata.layers || []) {
      const url = descriptorUrl(layer.preview);
      if (!url) continue;
      this.loadImage(url, (image) => {
        this.cutouts.set(layer.socket, image);
        this.requestDraw();
      }, () => generation === this.metadataGeneration);
    }
    this.backgroundSource = null;
    this.resolveBackground(true);
    this.requestDraw();
  }

  upstreamUpdated(nodeId) {
    const sourceNames = (this.node.comfyClass === "UC_StagedLayeredBackgroundComposite" || this.node.type === "UC_StagedLayeredBackgroundComposite")
      ? ["background", "staged_foregrounds"]
      : ["background", ...this.connectedLayers()];
    const origins = sourceNames.map((name) => this.inputOrigin(name));
    if (!origins.some((origin) => String(origin?.link?.origin_id) === String(nodeId))) return;
    if (sourceNames.slice(1).some((name) => String(this.inputOrigin(name)?.link?.origin_id) === String(nodeId))) {
      this.metadata = null;
      this.metadataSignature = null;
      this.cutouts.clear();
      this.metadataGeneration++;
    }
    this.backgroundSource = null;
    this.resolveBackground(true);
    this.requestDraw();
  }

  layerMetadata(key) {
    return this.metadata?.layers?.find((layer) => layer.socket === key);
  }

  layerAspect(key) {
    const metadata = this.layerMetadata(key);
    return metadata?.crop_width > 0 && metadata?.crop_height > 0
      ? metadata.crop_width / metadata.crop_height
      : 1;
  }

  layerPlacement(key) {
    return normalizePlacement(this.data.layers[key] || DEFAULT_PLACEMENT);
  }

  dimensions() {
    const exact = this.metadataSignature === this.semanticSignature() ? this.metadata?.background : null;
    const width = exact?.width || this.backgroundImage?.naturalWidth || this.backgroundImage?.width;
    const height = exact?.height || this.backgroundImage?.naturalHeight || this.backgroundImage?.height;
    return width > 0 && height > 0 ? { width, height } : null;
  }

  viewFor(width, height, dimensions, layers) {
    const basePaddingX = dimensions.width * 0.25;
    const basePaddingY = dimensions.height * 0.25;
    let left = -basePaddingX;
    let top = -basePaddingY;
    let right = dimensions.width + basePaddingX;
    let bottom = dimensions.height + basePaddingY;
    for (const key of layers) {
      const rect = this.rectFor(key, dimensions);
      left = Math.min(left, rect.x);
      top = Math.min(top, rect.y);
      right = Math.max(right, rect.x + rect.width);
      bottom = Math.max(bottom, rect.y + rect.height);
    }
    const outerPadding = Math.max(right - left, bottom - top) * 0.04;
    left -= outerPadding;
    top -= outerPadding;
    right += outerPadding;
    bottom += outerPadding;
    const padding = 8;
    const scale = Math.min((width - padding * 2) / (right - left), (height - padding * 2) / (bottom - top));
    return {
      x: (width - (right - left) * scale) / 2 - left * scale,
      y: (height - (bottom - top) * scale) / 2 - top * scale,
      width: dimensions.width * scale,
      height: dimensions.height * scale,
      scale,
    };
  }

  requestDraw() {
    if (!this.drawFrame) this.drawFrame = requestAnimationFrame(() => {
      this.drawFrame = 0;
      this.draw();
    });
  }

  draw() {
    const width = this.stage.clientWidth;
    const height = this.stage.clientHeight;
    if (width <= 0 || height <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(height * dpr);
    }
    const context = this.canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#17191d";
    context.fillRect(0, 0, width, height);
    const dimensions = this.dimensions();
    const layers = this.connectedLayers();
    if (!dimensions || !this.backgroundImage) {
      this.view = null;
      this.status.textContent = this.inputOrigin("background")
        ? "Background preview unavailable"
        : "Connect a background image";
      this.status.hidden = false;
      return;
    }
    this.view = this.gesture?.view || this.viewFor(width, height, dimensions, layers);
    context.globalAlpha = 0.72;
    context.drawImage(this.backgroundImage, this.view.x, this.view.y, this.view.width, this.view.height);
    context.globalAlpha = 1;
    for (const key of layers) this.drawLayer(context, key, dimensions);
    context.lineWidth = 2;
    context.strokeStyle = "rgba(255,255,255,.65)";
    context.strokeRect(this.view.x, this.view.y, this.view.width, this.view.height);
    if (this.selected) this.drawHandles(context, this.selected, dimensions);
    const pending = layers.filter((key) => !this.layerMetadata(key)).length;
    this.status.textContent = pending
      ? `${pending} foreground${pending === 1 ? "" : "s"} pending removal pass`
      : `${layers.length} layer${layers.length === 1 ? "" : "s"} • back → front`;
    this.status.hidden = false;
  }

  rectFor(key, dimensions) {
    return placementToRect(dimensions.width, dimensions.height, this.layerAspect(key), this.layerPlacement(key));
  }

  toCanvasRect(rect) {
    return {
      x: this.view.x + rect.x * this.view.scale,
      y: this.view.y + rect.y * this.view.scale,
      width: rect.width * this.view.scale,
      height: rect.height * this.view.scale,
    };
  }

  drawLayer(context, key, dimensions) {
    const rect = this.toCanvasRect(this.rectFor(key, dimensions));
    const cutout = this.cutouts.get(key);
    if (cutout) context.drawImage(cutout, rect.x, rect.y, rect.width, rect.height);
    context.fillStyle = key === this.selected ? "rgba(64,180,255,.16)" : "rgba(255,255,255,.055)";
    context.fillRect(rect.x, rect.y, rect.width, rect.height);
    context.save();
    context.setLineDash(this.layerMetadata(key) ? [] : [6, 4]);
    context.lineWidth = key === this.selected ? 4 : 3;
    context.strokeStyle = "rgba(0,0,0,.8)";
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    context.lineWidth = key === this.selected ? 2 : 1;
    context.strokeStyle = key === this.selected ? "#65c9ff" : "rgba(255,255,255,.88)";
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    context.restore();
  }

  handlePoints(key, dimensions) {
    const rect = this.toCanvasRect(this.rectFor(key, dimensions));
    const left = rect.x;
    const right = rect.x + rect.width;
    const top = rect.y;
    const bottom = rect.y + rect.height;
    return { nw: { x: left, y: top }, ne: { x: right, y: top }, sw: { x: left, y: bottom }, se: { x: right, y: bottom } };
  }

  drawHandles(context, key, dimensions) {
    for (const point of Object.values(this.handlePoints(key, dimensions))) {
      context.fillStyle = "#111";
      context.fillRect(point.x - 5, point.y - 5, 10, 10);
      context.fillStyle = "#fff";
      context.fillRect(point.x - 3, point.y - 3, 6, 6);
    }
  }

  canvasPoint(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return {
      x: (event.clientX - bounds.left) * (this.canvas.clientWidth / bounds.width),
      y: (event.clientY - bounds.top) * (this.canvas.clientHeight / bounds.height),
    };
  }

  backgroundPoint(canvasPoint) {
    return { x: (canvasPoint.x - this.view.x) / this.view.scale, y: (canvasPoint.y - this.view.y) / this.view.scale };
  }

  pointerDown(event) {
    const dimensions = this.dimensions();
    const layers = this.connectedLayers();
    if (!this.view || !dimensions || !layers.length || event.button !== 0) return;
    event.preventDefault();
    this.canvas.focus();
    const canvasPoint = this.canvasPoint(event);
    const backgroundPoint = this.backgroundPoint(canvasPoint);
    let action = null;
    let handle = null;
    if (this.selected) {
      for (const [name, point] of Object.entries(this.handlePoints(this.selected, dimensions))) {
        if (Math.hypot(canvasPoint.x - point.x, canvasPoint.y - point.y) <= 12) {
          action = "resize";
          handle = name;
          break;
        }
      }
    }
    if (!action) {
      for (const key of [...layers].reverse()) {
        const rect = this.rectFor(key, dimensions);
        if (backgroundPoint.x >= rect.x && backgroundPoint.x <= rect.x + rect.width && backgroundPoint.y >= rect.y && backgroundPoint.y <= rect.y + rect.height) {
          this.selectLayer(key);
          action = "move";
          break;
        }
      }
    }
    if (!action) {
      const inside = backgroundPoint.x >= 0 && backgroundPoint.x <= dimensions.width && backgroundPoint.y >= 0 && backgroundPoint.y <= dimensions.height;
      if (!inside || !this.selected) return;
      action = "draw";
    }
    this.flushPlacement();
    this.node.graph?.beforeChange?.();
    this.gesture = {
      action,
      handle,
      key: this.selected,
      startCanvas: canvasPoint,
      startPoint: backgroundPoint,
      startRect: this.rectFor(this.selected, dimensions),
      originalValue: this.placementWidget.value,
      pointerId: event.pointerId,
      changed: false,
      view: { ...this.view },
    };
    this.canvas.setPointerCapture(event.pointerId);
  }

  pointerMove(event) {
    if (!this.gesture || !this.view) return;
    event.preventDefault();
    const dimensions = this.dimensions();
    const canvasPoint = this.canvasPoint(event);
    const point = this.backgroundPoint(canvasPoint);
    const deltaX = point.x - this.gesture.startPoint.x;
    const deltaY = point.y - this.gesture.startPoint.y;
    let rect;
    if (this.gesture.action === "move") {
      rect = moveRect(dimensions.width, dimensions.height, this.gesture.startRect, deltaX, deltaY);
    } else if (this.gesture.action === "resize") {
      const shortest = Math.min(dimensions.width, dimensions.height);
      rect = resizeRectFromDelta(
        this.gesture.startRect,
        this.gesture.handle,
        deltaX,
        deltaY,
        this.layerAspect(this.gesture.key),
        shortest * 0.05,
        shortest * 10,
      );
    } else {
      rect = drawRect(this.gesture.startPoint, point, this.layerAspect(this.gesture.key));
      if (Math.hypot(canvasPoint.x - this.gesture.startCanvas.x, canvasPoint.y - this.gesture.startCanvas.y) < 6) return;
      if (Math.max(rect.width, rect.height) < Math.min(dimensions.width, dimensions.height) * 0.05) return;
    }
    this.gesture.changed = true;
    this.queuePlacement(this.gesture.key, rectToPlacement(
      dimensions.width,
      dimensions.height,
      rect,
      this.layerPlacement(this.gesture.key),
    ));
  }

  pointerEnd(event, cancelled) {
    if (!this.gesture) return;
    if (cancelled || (this.gesture.action === "draw" && !this.gesture.changed)) {
      this.pendingPlacement = null;
      if (this.commitFrame) cancelAnimationFrame(this.commitFrame);
      this.commitFrame = 0;
      this.placementWidget.value = this.gesture.originalValue;
      this.data = parsePlacementData(this.gesture.originalValue);
      this.placementWidget.callback?.(this.placementWidget.value, app.canvas, this.node);
    } else {
      this.flushPlacement();
    }
    const pointerId = Number.isInteger(event.pointerId) ? event.pointerId : this.gesture.pointerId;
    if (Number.isInteger(pointerId) && this.canvas.hasPointerCapture?.(pointerId)) {
      this.canvas.releasePointerCapture(pointerId);
    }
    this.gesture = null;
    this.node.graph?.afterChange?.();
    this.syncNumericControls();
    this.requestDraw();
  }

  queuePlacement(key, placement) {
    this.pendingPlacement = { key, placement };
    if (!this.commitFrame) this.commitFrame = requestAnimationFrame(() => this.flushPlacement());
  }

  flushPlacement() {
    if (this.commitFrame) cancelAnimationFrame(this.commitFrame);
    this.commitFrame = 0;
    if (!this.pendingPlacement) return;
    const { key, placement } = this.pendingPlacement;
    this.pendingPlacement = null;
    this.data.layers[key] = normalizePlacement(placement);
    this.placementWidget.value = serializePlacementData(this.data);
    this.placementWidget.callback?.(this.placementWidget.value, app.canvas, this.node);
    this.node.graph?.setDirtyCanvas?.(true, true);
    this.syncNumericControls();
    this.requestDraw();
  }

  selectLayer(key) {
    if (!this.connectedLayers().includes(key)) return;
    this.selected = key;
    this.layerSelect.value = key;
    this.syncNumericControls();
    this.requestDraw();
  }

  syncNumericControls() {
    const placement = this.selected ? this.layerPlacement(this.selected) : DEFAULT_PLACEMENT;
    for (const [field, input] of Object.entries(this.inputs)) {
      input.disabled = !this.selected;
      input.value = Number(placement[field]).toFixed(4);
    }
    this.resetButton.disabled = !this.selected;
  }

  numericChanged(field, value) {
    if (!this.selected) return;
    const placement = { ...this.layerPlacement(this.selected), [field]: Number(value) };
    this.node.graph?.beforeChange?.();
    this.queuePlacement(this.selected, placement);
    this.flushPlacement();
    this.node.graph?.afterChange?.();
  }

  resetSelected() {
    if (!this.selected) return;
    this.node.graph?.beforeChange?.();
    this.queuePlacement(this.selected, DEFAULT_PLACEMENT);
    this.flushPlacement();
    this.node.graph?.afterChange?.();
  }

  keyDown(event) {
    if (!this.selected || !this.view) return;
    if (event.key === "Escape" && this.gesture) {
      event.preventDefault();
      this.pointerEnd({ pointerId: event.pointerId }, true);
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      event.stopPropagation();
      this.resetSelected();
      return;
    }
    const deltas = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (!deltas[event.key]) return;
    event.preventDefault();
    event.stopPropagation();
    if (!this.keyTransaction) {
      this.node.graph?.beforeChange?.();
      this.keyTransaction = true;
    }
    const dimensions = this.dimensions();
    const amount = event.shiftKey ? 10 : 1;
    const [dx, dy] = deltas[event.key];
    const rect = moveRect(
      dimensions.width,
      dimensions.height,
      this.rectFor(this.selected, dimensions),
      dx * amount,
      dy * amount,
    );
    this.queuePlacement(this.selected, rectToPlacement(
      dimensions.width,
      dimensions.height,
      rect,
      this.layerPlacement(this.selected),
    ));
  }

  finishKeyTransaction() {
    if (!this.keyTransaction) return;
    this.flushPlacement();
    this.keyTransaction = false;
    this.node.graph?.afterChange?.();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    clearInterval(this.pollTimer);
    this.resizeObserver?.disconnect();
    if (this.drawFrame) cancelAnimationFrame(this.drawFrame);
    if (this.commitFrame) cancelAnimationFrame(this.commitFrame);
    editors.delete(this);
  }
}

function install(node) {
  if (node.__ucLayeredPlacementEditor) return node.__ucLayeredPlacementEditor;
  const editor = new LayeredPlacementEditor(node);
  node.__ucLayeredPlacementEditor = editor;
  editors.add(editor);
  const output = latestOutputs[String(node.id)];
  if (output) editor.setOutput(output);
  return editor;
}

app.registerExtension({
  name: "UtilsCollection.LayeredBackgroundEditor",
  nodeCreated(node) {
    if (NODE_TYPES.has(node.comfyClass) || NODE_TYPES.has(node.type)) install(node);
  },
  loadedGraphNode(node) {
    if (NODE_TYPES.has(node.comfyClass) || NODE_TYPES.has(node.type)) install(node).ensureSize();
  },
  onNodeOutputsUpdated(outputs) {
    latestOutputs = { ...latestOutputs, ...outputs };
    for (const [nodeId] of Object.entries(outputs || {})) {
      for (const editor of editors.values()) editor.upstreamUpdated(nodeId);
    }
    for (const [nodeId, output] of Object.entries(outputs || {})) {
      for (const editor of editors.values()) {
        if (String(editor.node.id) === String(nodeId)) editor.setOutput(output);
      }
    }
  },
});
