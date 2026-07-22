export const DEFAULT_BACKGROUND_MODEL = "birefnet";

export function normalizeBackgroundModel(value) {
  return value === "lucida" ? "lucida" : DEFAULT_BACKGROUND_MODEL;
}

export function toggleBackgroundModel(value) {
  return normalizeBackgroundModel(value) === "birefnet" ? "lucida" : "birefnet";
}

export function backgroundModelAppearance(value) {
  if (normalizeBackgroundModel(value) === "lucida") {
    return {
      text: "Lucida",
      border: "rgba(255,120,205,.85)",
      background: "rgba(158,38,119,.90)",
    };
  }
  return {
    text: "BiRefNet",
    border: "rgba(101,201,255,.8)",
    background: "rgba(25,105,145,.88)",
  };
}
