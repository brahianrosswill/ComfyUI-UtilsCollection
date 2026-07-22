import test from "node:test";
import assert from "node:assert/strict";

import { buildOutputClosure, buildStagingPrompt } from "../web/staged_queue.js";

test("staging prompt keeps only the target and recursive upstream closure", () => {
  const output = {
    1: { inputs: { value: 1 }, class_type: "Source" },
    2: { inputs: { image: ["1", 0] }, class_type: "Foreground" },
    3: {
      inputs: {
        background: ["1", 0],
        foreground_0: ["2", 0],
        execution_mode: "run_staged",
      },
      class_type: "UC_StagedLayeredBackgroundComposite",
    },
    4: { inputs: { image: ["1", 0] }, class_type: "UnrelatedOutput" },
  };

  const prompt = buildStagingPrompt({ output, workflow: { stable: true } }, 3);

  assert.deepEqual(Object.keys(prompt.output).sort(), ["1", "2", "3"]);
  assert.equal(prompt.output[3].inputs.execution_mode, "run_staging");
  assert.equal(prompt.workflow.stable, true);
  assert.equal(output[3].inputs.execution_mode, "run_staged");
  assert.notEqual(prompt.output[3], output[3]);
  assert.notEqual(prompt.output[3].inputs, output[3].inputs);
});

test("output closure preserves primitive and literal-array inputs", () => {
  const literal = [0.1, 0.2, 0.3];
  const output = {
    target: { inputs: { literal, execution_mode: "full_run" }, class_type: "Target" },
  };
  const filtered = buildOutputClosure(output, "target", { execution_mode: "run_staging" });
  assert.equal(filtered.target.inputs.literal, literal);
  assert.equal(filtered.target.inputs.execution_mode, "run_staging");
});

test("missing staging target fails clearly", () => {
  assert.throws(
    () => buildStagingPrompt({ output: {} }, "missing"),
    /prompt node missing is missing/,
  );
});
