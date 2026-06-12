import json
import os
from comfy_api.latest import io, ComfyAPI

api = ComfyAPI()

async def register_replacements():
    """Register all node replacements for this package using mapping data."""
    mapping_path = os.path.join(os.path.dirname(__file__), "plans", "widget_mapping_audit.json")

    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
    except Exception as e:
        print(f"[ComfyUI-UtilsCollection] Error loading widget_mapping_audit.json: {e}")
        return

    for new_node_id, data in mappings.items():
        try:
            old_node_id = data.get("old_node_id")
            old_widget_ids = data.get("old_widget_ids")

            if not old_node_id:
                continue

            await api.node_replacement.register(io.NodeReplace(
                new_node_id=new_node_id,
                old_node_id=old_node_id,
                old_widget_ids=old_widget_ids
            ))
        except Exception as e:
            print(f"[ComfyUI-UtilsCollection] Failed to register replacement for {new_node_id}: {e}")
