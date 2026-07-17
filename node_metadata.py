import re


DESCRIPTIONS = {
    "UC_AdjustedResolutionParameters": "Calculates aligned base and upscaled image dimensions from width, height, scale, and multiple.",
    "UC_ResolutionSelectorExtended": "Calculates width and height from an aspect ratio and target megapixel count.",
    "UC_ImageScaleAndResolutionPicker": "Resizes an image to a megapixel target and returns base and upscaled dimensions.",
    "UC_Image_Color_Noise": "Adds configurable color noise to an image.",
    "UC_LoadImagePath": "Loads an image and mask from an explicit filesystem path.",
    "UC_LoadImageDirectory": "Loads images from a directory for batch or iterative workflows.",
    "UC_ListToImageBatch": "Combines a list of compatible images into one image batch.",
    "UC_ImageMatchProperties": "Adjusts an image to match the size and properties of a reference image.",
    "UC_OpticalFlowComposite": "Composites images using motion estimated with optical flow.",
    "UC_ImageInwardEdgeFill": "Fills image edges inward to extend surrounding content.",
    "UC_ImageIterativeStretchFill": "Fills image borders by iteratively stretching nearby pixels.",
    "UC_TextOverlayNode": "Draws configurable text over an image.",
    "UC_ModifyMask": "Expands, contracts, blurs, or otherwise adjusts a mask.",
    "UC_ImageBlendByMask": "Blends two images using a mask and configurable blend behavior.",
    "UC_ImagePad": "Pads an image and produces the corresponding placement mask.",
    "UC_CropByMask": "Crops an image and mask to the nonzero mask bounds.",
    "UC_ImageCropMerge": "Merges a processed crop back into its original image region.",
    "UC_ImageAndMaskResize": "Resizes an image and mask together to explicit or reference dimensions.",
    "UC_ResizeMask": "Resizes a mask with selectable interpolation and crop behavior.",
    "UC_MediaPipeFaceCompositeOptions": "Configures landmark regions and blending options for MediaPipe face compositing.",
    "UC_MediaPipeFaceComposite": "Detects face landmarks and composites selected facial regions between images.",
    "UC_SystemMessagePresets": "Provides reusable system-message presets for image prompting.",
    "UC_SystemMessageVideoPresets": "Provides reusable system-message presets for video prompting.",
    "UC_InstructPromptPresets": "Provides reusable image instruction prompt presets.",
    "UC_InstructPromptVideoPresets": "Provides reusable video instruction prompt presets.",
    "UC_BonusPromptPresets": "Provides optional prompt enhancement presets for images.",
    "UC_BonusPromptVideoPresets": "Provides optional prompt enhancement presets for videos.",
    "UC_LegacyPromptPresets": "Provides older prompt presets retained for workflow compatibility.",
    "UC_EditTargetPresets": "Provides preset descriptions of image regions or subjects to edit.",
    "UC_EditOpPresets": "Provides preset image editing operations and instructions.",
    "UC_CameraShotPresets": "Provides camera framing, angle, and shot presets.",
    "UC_UnifiedPresets": "Combines multiple prompt preset families into one selector.",
    "UC_VLMSysInstrPresets": "Provides system instruction presets for vision-language models.",
    "UC_VLMSysQueryAddPresets": "Provides supplemental query presets for vision-language models.",
    "UC_VLMSysInstrAdvPresets": "Provides advanced system instruction presets for vision-language models.",
    "UC_AttentionBiasTextEncode": "Encodes text while applying token-level attention bias controls.",
    "UC_TextConsensusBlendConfig": "Configures consensus-based blending of multiple text conditioning tensors.",
    "UC_VisualFusionConfig": "Configures grid-aware fusion of visual token embeddings from multiple images.",
    "UC_ConditioningConsensusBlend": "Blends multiple conditioning outputs after encoding while preserving reference placement.",
    "UC_TextEncodeLtxv2SystemPrompt": "Encodes LTXV2 text with a custom system prompt.",
    "UC_TextEncodeSystemPrompt": "Encodes text with a custom system prompt using the connected text encoder.",
    "UC_WeightedTextEncodeSystemPrompt": "Encodes weighted text with a custom system prompt.",
    "UC_TextEncodeSystemEditAdvanced": "Encodes advanced image-edit conditioning with custom prompts and multiple images.",
    "UC_TextEncodeGemmaSystemEditAdvanced": "Encodes advanced Gemma image-edit conditioning with custom system prompts.",
    "UC_AdvancedVisualConditioningEncode": "Encodes and fuses visual conditioning from multiple images with advanced controls.",
    "UC_VLMInputEmbeds": "Exports raw input embeddings from a supported vision-language model encoder.",
    "UC_Krea2TokenAttentionWeight": "Applies phrase-level attention weights while encoding Krea2 visual conditioning.",
    "UC_TextGenerate": "Generates text with a connected language or vision-language model.",
    "UC_TextGenerateQwen35SystemPrompt": "Generates Qwen3.5 text with optional image input and a custom system message.",
    "UC_SwitchInverseNode": "Selects between two inputs using an inverted boolean switch.",
    "UC_SoftSwitchInverseNode": "Selects available inputs using an inverted soft switch.",
    "UC_IntegerRangeRandom": "Returns a seeded random integer within a configurable range.",
    "UC_TagNormalizeCombine": "Normalizes, deduplicates, scores, and combines tag strings.",
    "UC_RandInt": "Outputs an integer widget with generation-time control behavior.",
    "UC_StaticInt": "Outputs one reusable static integer value.",
    "UC_StaticFloat": "Outputs one reusable static floating-point value with 0.01 increments.",
    "UC_RandIntRange": "Returns a deterministic random integer between minimum and maximum values.",
    "UC_ColorConvertNode": "Converts colors between picker, hexadecimal, integer, and RGB string formats.",
    "UC_ExtractBoundingBox": "Extracts x, y, width, and height from a bounding-box value.",
    "UC_AdjustBoundingBox": "Adjusts bounding-box position and dimensions with boundary controls.",
    "UC_Krea2LayerProbe": "Measures and optionally saves Krea2 conditioning-layer activation statistics.",
    "UC_Krea2LayerAblator": "Removes selected refusal-direction components from Krea2 conditioning layers.",
    "UC_EncoderNodesGuide": "Returns documentation for the node pack's advanced encoder workflows.",
    "UC_LoraLoaderCLIPOnly": "Loads a LoRA into the text encoder without modifying the diffusion model.",
    "UC_BoldFrakturTextStyle": "Converts supported text characters to bold Fraktur Unicode styling.",
    "UC_UnBoldFrakturTextStyle": "Converts bold Fraktur Unicode characters back to plain text.",
    "UC_WordJoiner": "Joins words with Unicode word-joiner characters.",
    "UC_UnWordJoiner": "Removes Unicode word-joiner characters from text.",
    "UC_JSONMinifyRepair": "Repairs common JSON formatting issues and returns compact JSON text.",
    "UC_StringUnescape": "Converts escaped character sequences into their literal string values.",
    "Ideogram4SchedulerPreset": "Provides scheduler and sampling parameters tuned for Ideogram 4 workflows.",
}


EXTRA_ALIASES = {
    "UC_StaticInt": ["primitive integer", "number", "shared integer", "constant int"],
    "UC_StaticFloat": ["primitive float", "number", "decimal", "megapixel", "shared value", "constant float"],
    "UC_ConditioningConsensusBlend": ["conditioning merge", "conditioning combine", "post encoder", "cwb"],
    "UC_VisualFusionConfig": ["image token fusion", "visual blend", "dither", "checkerboard"],
    "UC_TextConsensusBlendConfig": ["conditioning blend config", "text merge", "cwb config"],
    "UC_MediaPipeFaceComposite": ["face swap", "face landmarks", "face blend", "mediapipe"],
    "UC_MediaPipeFaceCompositeOptions": ["face regions", "face landmarks", "mediapipe options"],
    "UC_ResolutionSelectorExtended": ["megapixels", "aspect ratio", "width height", "resolution"],
    "UC_ImageScaleAndResolutionPicker": ["megapixels", "image resize", "upscale", "resolution"],
    "UC_LoadImagePath": ["load image", "image path", "absolute path"],
    "UC_LoadImageDirectory": ["image folder", "batch loader", "directory loader"],
    "UC_TextGenerate": ["llm", "vlm", "chat", "text generation"],
    "UC_TextGenerateQwen35SystemPrompt": ["llm", "vlm", "qwen", "chat", "system prompt"],
    "UC_VLMInputEmbeds": ["embedding export", "visual embeddings", "qwen embeddings", "krea embeddings"],
    "UC_LoraLoaderCLIPOnly": ["clip lora", "text encoder lora", "load lora"],
}


def _search_terms(schema):
    text = f"{schema.node_id} {schema.display_name or ''} {schema.category or ''}"
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", text)
    ignored = {"uc", "node", "utils", "utilities", "advanced"}
    terms = [word.lower() for word in words if word.lower() not in ignored]
    display = (schema.display_name or "").strip().lower()
    if display:
        terms.append(display)
    return terms


def enrich_node_metadata(node_class):
    if node_class.__dict__.get("_uc_metadata_enriched", False):
        return
    original = node_class.define_schema.__func__

    @classmethod
    def define_schema(cls):
        schema = original(cls)
        if cls is not node_class or schema.is_deprecated or "(Legacy)" in (schema.display_name or ""):
            return schema
        description = DESCRIPTIONS.get(schema.node_id)
        if not schema.description:
            schema.description = description or f"Provides {schema.display_name or schema.node_id} functionality."
        aliases = list(schema.search_aliases or [])
        aliases.extend(_search_terms(schema))
        aliases.extend(EXTRA_ALIASES.get(schema.node_id, []))
        schema.search_aliases = list(dict.fromkeys(alias for alias in aliases if alias))
        return schema

    node_class.define_schema = define_schema
    node_class._uc_metadata_enriched = True


def enrich_node_list(node_classes):
    for node_class in node_classes:
        enrich_node_metadata(node_class)
    return node_classes
