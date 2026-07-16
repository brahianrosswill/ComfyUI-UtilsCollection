from comfy_api.latest import ComfyAPI, io


api = ComfyAPI()


# These migrations only cover nodes whose canonical replacements retain the
# same input and output IDs. More specialized wrappers remain loadable in 0.10
# and are deliberately not auto-rewritten with guessed widget mappings.
REPLACEMENTS = (
    (
        "UC_TextEncodeSystemEditAdvanced",
        "TextEncodeSystemEditPlusAdvanced",
        ["prompt", "system_prompt", "vlm_resolution", "vae_resolution", "ref_latent_mode"],
    ),
    (
        "UC_TextEncodeGemmaSystemEditAdvanced",
        "TextEncodeGemmaSystemEditPlusAdvanced",
        ["prompt", "system_prompt", "vlm_resolution", "vae_resolution", "ref_latent_mode"],
    ),
    (
        "UC_AdvancedVisualConditioningEncode",
        "TextEncodeKrea2SystemEditScaledAdv",
        [
            "prompt",
            "system_prompt",
            "vlm_resolution",
            "formula",
            "padding_method",
            "vae_resolution",
            "ref_latent_mode",
            "multiplier",
        ],
    ),
    (
        "UC_VLMInputEmbeds",
        "UC_Qwen3VLInputEmbeds",
        ["prompt", "image_paths", "vlm_resolution", "file_names", "slice_visual_tokens"],
    ),
    (
        "UC_VLMInputEmbeds",
        "UC_Krea2InputEmbeds",
        ["prompt", "image_paths", "vlm_resolution", "file_names", "slice_visual_tokens"],
    ),
    (
        "UC_Krea2TokenAttentionWeight",
        "TextEncodeKrea2SysEditScaledAdvAttn",
        [
            "prompt",
            "system_prompt",
            "attention_weights",
            "vlm_resolution",
            "strength",
            "formula",
            "padding_method",
            "vae_resolution",
            "ref_latent_mode",
            "multiplier",
        ],
    ),
)


async def register_replacements():
    """Register tracked, interface-preserving node migrations."""
    for new_node_id, old_node_id, old_widget_ids in REPLACEMENTS:
        try:
            await api.node_replacement.register(
                io.NodeReplace(
                    new_node_id=new_node_id,
                    old_node_id=old_node_id,
                    old_widget_ids=old_widget_ids,
                )
            )
        except Exception as exc:
            print(
                f"[ComfyUI-UtilsCollection] Failed to register replacement "
                f"{old_node_id} -> {new_node_id}: {exc}"
            )
