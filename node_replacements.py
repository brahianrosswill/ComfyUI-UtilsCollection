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
    ("UC_LogicIF", "LogicIF", ["if_condition"]),
    ("UC_LogicAND", "LogicAND", None),
    ("UC_LogicOR", "LogicOR", None),
    ("UC_LogicNOT", "LogicNOT", ["input"]),
    ("UC_LogicXOR", "LogicXOR", None),
    ("UC_MathAdd", "MathAdd", None),
    ("UC_MathSubtract", "MathSubtract", None),
    ("UC_MathMultiply", "MathMultiply", None),
    ("UC_MathDivide", "MathDivide", ["handle_zero"]),
    ("UC_MathPower", "MathPower", None),
    ("UC_MathFloor", "MathFloor", None),
    ("UC_MathCeil", "MathCeil", None),
    ("UC_MathRound", "MathRound", ["decimals"]),
    ("UC_MathModulo", "MathModulo", None),
    ("UC_MathAbs", "MathAbs", None),
    ("UC_MathSqrt", "MathSqrt", None),
    ("UC_MathSin", "MathSin", ["unit"]),
    ("UC_MathCos", "MathCos", ["unit"]),
    ("UC_MathTan", "MathTan", ["unit"]),
    ("UC_MathMin", "MathMin", None),
    ("UC_MathMax", "MathMax", None),
    ("UC_MathClamp", "MathClamp", None),
    ("UC_MathNumberConvert", "MathNumberConvert", None),
    ("UC_StringToNumber", "StringToNumber", ["string"]),
    ("UC_NumberToString", "NumberToString", None),
    ("UC_MathCompare", "MathCompare", ["comparison"]),
    ("UC_MathOperation", "MathOperation", ["operation"]),
    ("UC_MathAspectRatio", "MathAspectRatio", ["width", "height"]),
    (
        "UC_SigmoidOffsetScheduler",
        "SigmoidOffsetScheduler",
        ["steps", "square_k", "base_c", "start_sigma"],
    ),
)

MAPPED_REPLACEMENTS = (
    (
        "UC_PowerShiftScheduler",
        "PowerShiftScheduler",
        [
            "steps",
            "power",
            "midpoint_shift",
            "discard_penultimate",
            "denoise",
        ],
        [
            {"new_id": "model", "old_id": "model"},
            {"new_id": "steps", "old_id": "steps"},
            {"new_id": "power", "old_id": "power"},
            {"new_id": "midpoint_shift", "old_id": "midpoint_shift"},
        ],
    ),
    (
        "UC_RadianceShiftScheduler",
        "RadianceShiftScheduler",
        [
            "steps",
            "power",
            "midpoint_shift",
            "discard_penultimate",
            "denoise",
        ],
        [
            {"new_id": "model", "old_id": "model"},
            {"new_id": "steps", "old_id": "steps"},
            {"new_id": "power", "old_id": "power"},
            {"new_id": "midpoint_shift", "old_id": "midpoint_shift"},
        ],
    ),
    (
        "UC_SigmaCurveFromPointsScheduler",
        "SigmaCurveFromPointsScheduler",
        ["steps", "discard_penultimate", "denoise", "custom_points"],
        [
            {"new_id": "steps", "old_id": "steps"},
            {"new_id": "custom_points", "old_id": "custom_points"},
        ],
    ),
    (
        "UC_SigmaCurvePchipScheduler",
        "SigmaCurvePchipScheduler",
        ["steps", "discard_penultimate", "denoise", "custom_points"],
        [
            {"new_id": "steps", "old_id": "steps"},
            {"new_id": "custom_points", "old_id": "custom_points"},
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
    for (
        new_node_id,
        old_node_id,
        old_widget_ids,
        input_mapping,
    ) in MAPPED_REPLACEMENTS:
        try:
            await api.node_replacement.register(
                io.NodeReplace(
                    new_node_id=new_node_id,
                    old_node_id=old_node_id,
                    old_widget_ids=old_widget_ids,
                    input_mapping=input_mapping,
                    output_mapping=[{"new_idx": 0, "old_idx": 0}],
                )
            )
        except Exception as exc:
            print(
                f"[ComfyUI-UtilsCollection] Failed to register replacement "
                f"{old_node_id} -> {new_node_id}: {exc}"
            )
