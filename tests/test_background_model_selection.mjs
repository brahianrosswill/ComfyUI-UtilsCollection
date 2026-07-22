import assert from "node:assert/strict";
import test from "node:test";

import {
  backgroundModelAppearance,
  normalizeBackgroundModel,
  toggleBackgroundModel,
} from "../web/background_model_selection.js";

test("background-removal selection defaults to BiRefNet and toggles", () => {
  assert.equal(normalizeBackgroundModel(undefined), "birefnet");
  assert.equal(normalizeBackgroundModel("unknown"), "birefnet");
  assert.equal(toggleBackgroundModel("birefnet"), "lucida");
  assert.equal(toggleBackgroundModel("lucida"), "birefnet");
});

test("background-removal selection exposes blue and pink button states", () => {
  const birefnet = backgroundModelAppearance("birefnet");
  const lucida = backgroundModelAppearance("lucida");
  assert.equal(birefnet.text, "BiRefNet");
  assert.match(birefnet.border, /101,201,255/);
  assert.equal(lucida.text, "Lucida");
  assert.match(lucida.border, /255,120,205/);
});
