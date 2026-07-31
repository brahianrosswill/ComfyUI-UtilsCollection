import test from "node:test";
import assert from "node:assert/strict";

import {
  hexToRgb,
  hslToRgb,
  normalizeHex,
  pickerOwnsTarget,
  rgbToHex,
  rgbToHsl,
} from "../web/paint_color_picker.js";

test("picker ownership excludes unrelated controls in same node", () => {
  const triggerTarget = {}, panelTarget = {}, otherControl = {};
  const trigger = { contains: (target) => target === triggerTarget };
  const panel = { contains: (target) => target === panelTarget };
  assert.equal(pickerOwnsTarget(trigger, panel, triggerTarget), true);
  assert.equal(pickerOwnsTarget(trigger, panel, panelTarget), true);
  assert.equal(pickerOwnsTarget(trigger, panel, otherControl), false);
});

test("hex validation accepts only complete six-digit colors", () => {
  assert.equal(normalizeHex("ABCDEF"), "#abcdef");
  assert.equal(normalizeHex("#012345"), "#012345");
  for (const invalid of ["#fff", "#12345g", "", null]) assert.equal(normalizeHex(invalid), null);
});

test("RGB and hex conversions preserve exact channel values", () => {
  assert.equal(rgbToHex(18, 52, 86), "#123456");
  assert.deepEqual(hexToRgb("#123456"), { r: 18, g: 52, b: 86 });
});

test("HSL round trips representative RGB colors", () => {
  for (const rgb of [
    { r: 255, g: 0, b: 0 }, { r: 12, g: 180, b: 92 }, { r: 77, g: 77, b: 77 },
  ]) {
    const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
    assert.deepEqual(hslToRgb(hsl.h, hsl.s, hsl.l), rgb);
  }
});
