from typing_extensions import override

# Monkey-patch ComfyUI core's Qwen3VLTokenizer.tokenize_with_weights on startup
# to fix a core bug where loaded embedding Tensors in the token list cause a
# "Boolean value of Tensor with more than one value is ambiguous" crash on line 170.
try:
    import comfy.text_encoders.qwen3vl
    import torch

    def safe_tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=False, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]

        skip_template = text.startswith('<|im_start|>')
        if prevent_empty_text and text == '':
            text = ' '

        if skip_template:
            llama_text = text
        else:
            if llama_template is not None:
                template = llama_template
            elif len(images) == 0:
                template = self.llama_template
            else:
                template = self.llama_template_images
                if len(images) > 1:
                    vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
                    template = template.replace(vision_block, vision_block * len(images), 1)
            llama_text = template.format(text)
            if not thinking:
                llama_text += "<think>\n\n</think>\n\n"

        tokens = super(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer, self).tokenize_with_weights(
            llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs
        )
        key_name = next(iter(tokens))
        embed_count = 0
        for r in tokens[key_name]:
            for i in range(len(r)):
                # Core Bug Fix: Safely check type of r[i][0] before comparing to integer 151655
                if isinstance(r[i][0], (int, float)) and r[i][0] == 151655:  # <|image_pad|>
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens

    comfy.text_encoders.qwen3vl.Qwen3VLTokenizer.tokenize_with_weights = safe_tokenize_with_weights
    print("[ComfyUI-UtilsCollection] Successfully monkey-patched Qwen3VLTokenizer core tokenizer with safe embedding checks.")
except Exception as e:
    print(f"[ComfyUI-UtilsCollection] Failed to monkey-patch Qwen3VLTokenizer core tokenizer: {e}")

from .encoder_nodes import *
from .image_nodes import *
from .preset_nodes import *
from .vlm_nodes import *
from .parameter_nodes import *
from .utils_nodes import *
from .legacy_nodes import *
from .scheduler_nodes import *

from comfy_api.latest import ComfyExtension, io
from .node_replacements import register_replacements

class SamplingUtils(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UC_AdjustedResolutionParameters,
            UC_ResolutionSelectorExtended,
            UC_ImageScaleAndResolutionPicker,
            UC_Image_Color_Noise,
            UC_TextEncodeFlux2SystemPrompt,
            UC_TextEncodeKleinSystemPrompt,
            UC_TextEncodeKrea2SystemPrompt,
            UC_TextEncodeLtxv2SystemPrompt,
            UC_TextEncodeZITSystemPrompt,
            UC_TextEncodeZImageThinkPrompt,
            UC_TextEncodeSystemPrompt,
            UC_ScaledBiasTextEncodeFlux2SystemPrompt,
            UC_ScaledBiasTextEncodeKleinSystemPrompt,
            UC_ScaledBiasTextEncodeLtxv2SystemPrompt,
            UC_ScaledBiasTextEncodeZITSystemPrompt,
            UC_ScaledBiasTextEncodeZImageThinkPrompt,
            UC_ScaledBiasTextEncodeSystemPrompt,
            UC_ModifyMask,
            UC_ImageBlendByMask,
            UC_SystemMessagePresets,
            UC_SystemMessageVideoPresets,
            UC_InstructPromptPresets,
            UC_InstructPromptVideoPresets,
            UC_BonusPromptPresets,
            UC_BonusPromptVideoPresets,
            UC_EditTargetPresets,
            UC_EditOpPresets,
            UC_CameraShotPresets,
            UC_VLMSysInstrPresets,
            UC_VLMSysQueryAddPresets,
            UC_VLMSysInstrAdvPresets,
            UC_LegacyPromptPresets,
            UC_UnifiedPresets,
            UC_AttentionBiasTextEncode,
            UC_TagNormalizeCombine,
            UC_LoadImagePath,
            UC_LoadImageDirectory,
            UC_SwitchInverseNode,
            UC_SoftSwitchInverseNode,
            UC_IntegerRangeRandom,
            UC_ImageMatchPropertiesNode,
            UC_OpticalFlowComposite,
            UC_ImageInwardEdgeFill,
            UC_ImageIterativeStretchFill,
            UC_TextOverlayNode,
            UC_RandInt,
            UC_StaticInt,
            UC_RandIntRange,
            UC_TextGenerateQwen35SystemPrompt,
            UC_ColorConvertNode,
            UC_ExtractBoundingBox,
            UC_Krea2LayerProbe,
            UC_Krea2LayerAblator,
            UC_Krea2InputEmbeds,
            UC_EncoderNodesGuide,
            AdjustedResolutionParameters,
            ResolutionSelectorExtended,
            ImageScaleAndResolutionPicker,
            Image_Color_Noise,
            TextEncodeFlux2SystemPrompt,
            TextEncodeKleinSystemPrompt,
            TextEncodeLtxv2SystemPrompt,
            TextEncodeZITSystemPrompt,
            TextEncodeZImageThinkPrompt,
            TextEncodeSystemPrompt,
            ScaledBiasTextEncodeFlux2SystemPrompt,
            ScaledBiasTextEncodeKleinSystemPrompt,
            TextEncodeSystemEditPlus,
            TextEncodeSystemEditPlusAdvanced,
            TextEncodeKrea2SystemEditPlusAdvanced,
            TextEncodeEditPlusAdvanced,
            TextEncodeKrea2SystemEditScaledAdv,
            TextEncodeEditScaledAdv,
            TextEncodeGemmaSystemEditPlusAdvanced,
            ScaledBiasTextEncodeLtxv2SystemPrompt,
            ScaledBiasTextEncodeZITSystemPrompt,
            ScaledBiasTextEncodeZImageThinkPrompt,
            ScaledBiasTextEncodeSystemPrompt,
            ModifyMask,
            ImageBlendByMask,
            SystemMessagePresets,
            SystemMessageVideoPresets,
            InstructPromptPresets,
            InstructPromptVideoPresets,
            BonusPromptPresets,
            BonusPromptVideoPresets,
            EditTargetPresets,
            EditOpPresets,
            CameraShotPresets,
            VLMSysInstrPresets,
            VLMSysQueryAddPresets,
            VLMSysInstrAdvPresets,
            UnifiedPresets,
            AttentionBiasTextEncode,
            TagNormalizeCombine,
            SU_LoadImagePath,
            SU_LoadImageDirectory,
            SwitchInverseNode,
            SoftSwitchInverseNode,
            IntegerRangeRandom,
            ImageMatchPropertiesNode,
            OpticalFlowComposite,
            ImageInwardEdgeFill,
            ImageIterativeStretchFill,
            TextOverlayNode,
            RandInt,
            StaticInt,
            RandIntRange,
            TextGenerateQwen35SystemPrompt,
            ColorConvertNode,
            EncoderNodesGuide,
            Ideogram4SchedulerPreset,
        ]

    @override
    async def on_load(self) -> None:
        """Called by ComfyUI backend on startup to initialize resources and register API extensions."""
        await register_replacements()

async def comfy_entrypoint() -> SamplingUtils:
    return SamplingUtils()
