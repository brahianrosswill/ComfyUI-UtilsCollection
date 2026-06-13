from typing_extensions import override

from .encoder_nodes import *
from .image_nodes import *
from .preset_nodes import *
from .vlm_nodes import *
from .parameter_nodes import *
from .utils_nodes import *
from .legacy_nodes import *

from comfy_api.latest import ComfyExtension, io
from .node_replacements import register_replacements

class UtilsCollectionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UC_AdjustedResolutionParameters,
            UC_ResolutionSelectorExtended,
            UC_ImageScaleAndResolutionPicker,
            UC_Image_Color_Noise,
            UC_TextEncodeFlux2SystemPrompt,
            UC_TextEncodeKleinSystemPrompt,
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
        ]

    @override
    async def on_load(self) -> None:
        """Called by ComfyUI backend on startup to initialize resources and register API extensions."""
        await register_replacements()

async def comfy_entrypoint() -> UtilsCollectionExtension:
    return UtilsCollectionExtension()

from .legacy_nodes import comfy_entrypoint as legacy_comfy_entrypoint
from .legacy_nodes import SamplingUtils

async def comfy_entrypoint() -> SamplingUtils:
    return await legacy_comfy_entrypoint()
