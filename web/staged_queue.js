function addUpstreamNode(nodeId, source, filtered) {
  const id = String(nodeId);
  if (filtered[id]) return;
  const node = source[id];
  if (!node) throw new Error(`Cannot queue staged compositor: prompt node ${id} is missing.`);

  filtered[id] = { ...node, inputs: { ...(node.inputs || {}) } };
  for (const value of Object.values(node.inputs || {})) {
    if (!Array.isArray(value) || value.length < 2) continue;
    const upstreamId = String(value[0]);
    if (source[upstreamId]) addUpstreamNode(upstreamId, source, filtered);
  }
}

export function buildOutputClosure(output, targetNodeId, inputOverrides = {}) {
  const source = output || {};
  const targetId = String(targetNodeId);
  const filtered = {};
  addUpstreamNode(targetId, source, filtered);
  filtered[targetId] = {
    ...filtered[targetId],
    inputs: { ...filtered[targetId].inputs, ...inputOverrides },
  };
  return filtered;
}

export function buildStagingPrompt(prompt, targetNodeId) {
  if (!prompt?.output) throw new Error("Cannot queue staged compositor: prompt output is unavailable.");
  return {
    ...prompt,
    output: buildOutputClosure(prompt.output, targetNodeId, { execution_mode: "run_staging" }),
  };
}

export function updatePromptNodeInputs(prompt, updates) {
  if (!prompt?.output || !updates?.length) return prompt;
  const output = { ...prompt.output };
  let changed = false;
  for (const { nodeId, inputs } of updates) {
    const id = String(nodeId);
    if (!output[id]) continue;
    output[id] = { ...output[id], inputs: { ...(output[id].inputs || {}), ...inputs } };
    changed = true;
  }
  return changed ? { ...prompt, output } : prompt;
}
