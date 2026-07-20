# ComfyUI-UtilsCollection

A collection of ComfyUI nodes for modern text and multimodal conditioning, image and mask processing, prompt presets, workflow parameters, loading, and general utilities. The encoder nodes track current ComfyUI Core behavior while retaining compatible legacy node IDs where practical.

## Available nodes

The list below uses the canonical node IDs. Deprecated compatibility aliases remain registered for existing workflows but are not duplicated here.

### Text encoding and conditioning

- `UC_TextEncodeSystemPrompt`
- `UC_TextEncodeFlux2SystemPrompt`
- `UC_TextEncodeKleinSystemPrompt`
- `UC_TextEncodeKrea2SystemPrompt`
- `UC_TextEncodeLtxv2SystemPrompt`
- `UC_TextEncodeZITSystemPrompt`
- `UC_TextEncodeZImageThinkPrompt`
- `UC_WeightedTextEncodeSystemPrompt`
- `UC_ScaledBiasTextEncodeSystemPrompt`
- `UC_ScaledBiasTextEncodeFlux2SystemPrompt`
- `UC_ScaledBiasTextEncodeKleinSystemPrompt`
- `UC_ScaledBiasTextEncodeLtxv2SystemPrompt`
- `UC_ScaledBiasTextEncodeZITSystemPrompt`
- `UC_ScaledBiasTextEncodeZImageThinkPrompt`
- `UC_TextEncodeSystemEditAdvanced`
- `UC_TextEncodeGemmaSystemEditAdvanced`
- `UC_AdvancedVisualConditioningEncode`
- `UC_Krea2TokenAttentionWeight`
- `UC_AttentionBiasTextEncode`
- `UC_TextConsensusBlendConfig`
- `UC_VisualFusionConfig`
- `UC_ConditioningConsensusBlend`
- `UC_VLMInputEmbeds`
- `UC_Krea2LayerProbe`
- `UC_Krea2LayerAblator`
- `UC_EncoderNodesGuide`

### Image, mask, and compositing

- `UC_Image_Color_Noise`
- `UC_ModifyMask`
- `UC_ImageBlendByMask`
- `UC_ImagePad`
- `UC_CropByMask`
- `UC_ImageCropMerge`
- `UC_ImageAndMaskResize`
- `UC_ResizeMask`
- `UC_UnifiedBackgroundReplace`
- `UC_LayeredForegroundStage`
- `UC_StagedLayeredBackgroundComposite`
- `UC_LayeredBackgroundComposite`
- `UC_MediaPipeFaceCompositeOptions`
- `UC_MediaPipeFaceComposite`
- `UC_ListToImageBatch`
- `UC_ImageMatchPropertiesNode`
- `UC_OpticalFlowComposite`
- `UC_ImageInwardEdgeFill`
- `UC_ImageIterativeStretchFill`
- `UC_TextOverlayNode`

`UC_LayeredBackgroundComposite` builds one scene from a single background and ordered foreground sockets. In LiteGraph, queue it once to obtain exact background-removed cutouts, arrange each foreground with its own box and numeric placement controls, then queue again to render the final back-to-front composite. Each foreground socket accepts one image; `foreground_0` is the backmost layer.

The experimental staged workflow separates background removal from composition. Connect `UC_LayeredForegroundStage` to `UC_StagedLayeredBackgroundComposite`, leave the compositor's `use_staged` disabled, and queue once to create and retain the cutouts. Then enable `use_staged`, adjust placement, and queue again without evaluating the staging branch. Disable `use_staged` and queue to refresh from current foreground inputs. Retained cutouts belong to the compositor, live in server memory, and must be recreated after restarting ComfyUI.

### Resolution and workflow parameters

- `UC_AdjustedResolutionParameters`
- `UC_ResolutionSelectorExtended`
- `UC_ImageScaleAndResolutionPicker`
- `UC_SwitchInverseNode`
- `UC_SoftSwitchInverseNode`
- `UC_IntegerRangeRandom`
- `UC_RandInt`
- `UC_StaticInt`
- `UC_StaticFloat`
- `UC_RandIntRange`
- `UC_ColorConvertNode`
- `UC_ExtractBoundingBox`
- `UC_AdjustBoundingBox`

### Prompt presets

- `UC_SystemMessagePresets`
- `UC_SystemMessageVideoPresets`
- `UC_InstructPromptPresets`
- `UC_InstructPromptVideoPresets`
- `UC_BonusPromptPresets`
- `UC_BonusPromptVideoPresets`
- `UC_EditTargetPresets`
- `UC_EditOpPresets`
- `UC_CameraShotPresets`
- `UC_VLMSysInstrPresets`
- `UC_VLMSysQueryAddPresets`
- `UC_VLMSysInstrAdvPresets`
- `UC_LegacyPromptPresets`
- `UC_UnifiedPresets`

### Loading, text generation, and text utilities

- `UC_LoadImagePath`
- `UC_LoadImageDirectory`
- `UC_LoraLoaderCLIPOnly`
- `UC_TextGenerate`
- `UC_TextGenerateQwen35SystemPrompt`
- `UC_TagNormalizeCombine`
- `UC_BoldFrakturTextStyle`
- `UC_UnBoldFrakturTextStyle`
- `UC_WordJoiner`
- `UC_UnWordJoiner`
- `UC_JSONMinifyRepair`
- `UC_StringUnescape`

### Scheduler presets

- `Ideogram4SchedulerPreset`
