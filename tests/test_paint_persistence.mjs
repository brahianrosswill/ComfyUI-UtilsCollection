import test from "node:test";
import assert from "node:assert/strict";

import { paintAssetForNode } from "../web/paint_persistence.js";

test("paint asset remains stable for its owner and forks for a copied node", () => {
  const owned = paintAssetForNode({ asset_id: "stable", owner_node_id: "7", revision: 3 }, 7);
  assert.equal(owned.asset_id, "stable");
  assert.equal(owned.asset.filename, "uc-staged-paint-stable.png");
  const forked = paintAssetForNode(owned, 8);
  assert.notEqual(forked.asset_id, "stable");
  assert.equal(forked.owner_node_id, "8");
  assert.notEqual(forked.asset.filename, owned.asset.filename);
});
