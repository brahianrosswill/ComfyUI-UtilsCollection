from typing_extensions import override

from .encoder_nodes import *
from .image_nodes import *
from .preset_nodes import *
from .vlm_nodes import *
from .parameter_nodes import *
from .utils_nodes import *

from comfy_api.latest import ComfyExtension, io, ComfyAPI

# Initialize ComfyAPI instance
api = ComfyAPI()

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
        node_classes = await self.get_node_list()
        custom_mappings = {
            "UC_LoadImagePath": "SU_LoadImagePath",
            "UC_LoadImageDirectory": "SU_LoadImageDirectory",
            "UC_SwitchInverseNode": "ComfySwitchInverseNode",
            "UC_SoftSwitchInverseNode": "ComfySoftSwitchInverseNode",
            "UC_RandInt": "PrimitiveRandomInt",
            "UC_StaticInt": "PrimitiveStaticInt",
            "UC_RandIntRange": "PrimitiveRandomIntRange",
        }
        for node_class in node_classes:
            try:
                schema = node_class.define_schema()
                new_node_id = schema.node_id

                # Check if there is a custom mapping defined
                if new_node_id in custom_mappings:
                    old_node_id = custom_mappings[new_node_id]
                elif new_node_id.startswith("UC_"):
                    old_node_id = new_node_id[3:]  # Strip 'UC_'
                else:
                    continue  # Not a UC_ node and no custom mapping, skip

                await api.node_replacement.register(io.NodeReplace(
                    new_node_id=new_node_id,
                    old_node_id=old_node_id,
                ))
            except Exception as e:
                print(f"[ComfyUI-UtilsCollection] Failed to register replacement: {e}")

async def comfy_entrypoint() -> UtilsCollectionExtension:
    return UtilsCollectionExtension()
