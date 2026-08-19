# ComfyUI-UtilsCollection

A collection of ComfyUI nodes for modern text and multimodal conditioning, image and mask processing, prompt presets, workflow parameters, loading, and general utilities. The encoder nodes track current ComfyUI Core behavior while retaining compatible legacy node IDs where practical.

## Available nodes

The list below uses the canonical node IDs. Deprecated compatibility aliases remain registered for existing workflows but are not duplicated here.

### Text encoding and conditioning

- `UC_TextEncodeSystemPrompt`
- `UC_TextEncodeLtxv2SystemPrompt`
- `UC_WeightedTextEncodeSystemPrompt`
- `UC_TextEncodeSystemEditAdvanced`
- `UC_TextEncodeGemmaSystemEditAdvanced`
- `UC_AdvancedVisualConditioningEncode`
- `UC_AdvancedMiniMaxH3ImageToVideo`
- `UC_AdvancedMiniMaxH3ImageToVideoCombined`
- `UC_MiniMaxH3MediaConfig`
- `UC_MiniMaxH3FirstFrameReferences`
- `UC_AdvancedVisConEncoder`
- `UC_VisualConsensusConfiguration`
- `UC_AdvancedVisualConfiguration`
- `UC_AdvancedConsensusConfiguration`
- `UC_Krea2TokenAttentionWeight`
- `UC_AttentionBiasTextEncode`
- `UC_TextConsensusBlendConfig`
- `UC_VisualFusionConfig`
- `UC_ConditioningConsensusBlend`
- `UC_VLMInputEmbeds`
- `UC_Krea2LayerProbe`
- `UC_Krea2LayerAblator`
- `UC_EncoderNodesGuide`

#### UC_AdvancedMiniMaxH3ImageToVideo: Qwen-only 1024 VLM example

The optional MiniMax H3 Media Configurator adds ordered timestamped images as one Qwen Video timeline without VAE-encoding those images. Existing Pictures remain first and may still use Picture fusion; Video blocks are never fused. Optional audio is encoded by the connected MiniMax H3 audio VAE as native reference audio, while Qwen receives only an Audio label. User timestamp precision is preserved in Qwen's Video labels.

[Workflow JSON](workflows/UC_AdvancedMiniMaxH3ImageToVideo/QwenOnly_8Image_1024VLM_Workflow.json) | [API workflow JSON](workflows/UC_AdvancedMiniMaxH3ImageToVideo/QwenOnly_8Image_1024VLM_API.json) | [Workflow overview](workflows/UC_AdvancedMiniMaxH3ImageToVideo/QwenOnly_8Image_1024VLM_Overview.png) | [Reference images](https://github.com/silveroxides/ComfyUI-UtilsCollection/releases/download/advanced-minimax-h3-qwen-only-assets-v1/UC_AdvancedMiniMaxH3ImageToVideo_QwenOnly_8Image_1024VLM_References.zip) | [Turbo LoRA used](https://huggingface.co/silveroxides/MiniMax-H3_tests/blob/main/minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro.safetensors)

This example uses eight chronological storyboard frames as 1024-resolution Qwen3-VL/DeepStack references. The prompt associates each ordered `<Picture N>` entry with a target timestamp. With `ref_image_size` set to `none`, the images provide visual-token conditioning without VAE reference encoding.

The workflow demonstrates strong subject, composition, and approximate timeline control without a native reference video. Its eight images reproduced the main framing and progression of a 12.25-second source sequence in seven sampling steps on a 16 GB GPU. Picture timestamps are prompt instructions, not fixed frame anchors, so results remain stochastic.

Extract the separately hosted reference-image ZIP into `ComfyUI/input` before loading either workflow.

The workflow uses Core's **Create Video** and **Save Video** nodes and requires no other custom-node collection.

For headless use, start ComfyUI with its API reachable, extract the reference ZIP, then run:

```powershell
python workflows/UC_AdvancedMiniMaxH3ImageToVideo/run_api_workflow.py C:\path\to\reference-images
```

The standard-library runner uploads the eight images, substitutes the returned server filenames into the unchanged API workflow, queues it, waits for completion, and prints the saved-output metadata. Use `--server http://host:8188` for another ComfyUI server and `--seed N` to override the workflow seed.

<img src="workflows/UC_AdvancedMiniMaxH3ImageToVideo/QwenOnly_8Image_1024VLM_Overview.png" alt="UC Advanced MiniMax H3 Image to Video Qwen-only eight-image 1024 VLM workflow" width="1200">

#### Advanced visual consensus

`UC_AdvancedVisConEncoder` runs two sequential stages: it first constructs a
complete spatially fused conditioning independently at every selected VLM
resolution, then passes those complete conditionings through the same consensus
mathematics as `UC_ConditioningConsensusBlend`. Spatial fusion and consensus
are not alternatives and are never crossfaded.

Use `UC_VisualConsensusConfiguration` for the required joint configuration.
Its two activation Booleans remain authoritative. The visual method is always
selected there. A connected `UC_AdvancedVisualConfiguration` overrides the
duplicated block, dither, seed, encoder-path, cleanup, perturbation, and raw
export values. A connected `UC_AdvancedConsensusConfiguration` replaces the
named consensus preset and basal global scale with the complete custom
consensus configuration. Its authoritative preset selector defaults to
`custom`; named selections reuse their stored mathematics while advanced
position weighting, common-prefix preservation, global scale, and resolution
sampling remain available.

`block_size` is specific to block-interleave. `dither_ratio` and
`dither_pattern` are specific to random-dither. The simple joint node exposes
`resolution_samples` as a consensus control; Advanced Consensus Configuration
overrides it when connected. The configured value is exact, so `1` remains one
resolution sample regardless of visual-source or batch-lane count. Original VLM
resolution supports one sample but cannot construct adjacent resolution
variants.

A batch in the only connected image socket behaves like its images were
connected as separate visual sources. With multiple connected batched sockets,
equal indices form independent lanes, singleton sockets broadcast, and all
other batch lengths must match. Raw visual export uses the same spatial mask as
the base-resolution conditioning fusion.

### Image, mask, and compositing

- `UC_Image_Color_Noise`
- `UC_ExtractPrevalentColors`
- `UC_ModifyMask`
- `UC_ImageBlendByMask`
- `UC_ImagePad`
- `UC_CropByMask`
- `UC_StagedLayerCrops`
- `UC_ImageCropMerge`
- `UC_ExtractMask`
- `UC_ExtractImage`
- `UC_ImageAndMaskResize`
- `UC_ResizeMask`
- `UC_BackgroundRemovalPreserveAlpha`
- `UC_UnifiedBackgroundReplace`
- `UC_StagedLayeredBackgroundComposite`
- `UC_StagedIndividualComposites`
- `UC_StagedLayeredBackgroundCompositeOptions`
- `UC_StagedMediaPipeFaceBackgroundComposite`
- `UC_StagedMediaPipeFaceOptions`
- `UC_LayeredBackgroundComposite`
- `UC_MediaPipeFaceCompositeOptions`
- `UC_MediaPipeFaceComposite`
- `UC_ListToImageBatch`
- `UC_ImageMatchProperties`
- `UC_OpticalFlowComposite`
- `UC_ImageInwardEdgeFill`
- `UC_ImageIterativeStretchFill`
- `UC_TextOverlayNode`
- `UC_CompositeNodesGuide`

`UC_StagedLayeredBackgroundComposite` builds a scene from a background and ordered foreground sockets. Use `run_staging` to retain cutouts and populate the placement editor. Use `run_staged` to composite retained cutouts without loading models or evaluating foreground branches. Use `full_run` to restage and composite in one queue. `foreground_0` is the backmost layer. Retained cutouts are held in server memory and must be recreated after restarting ComfyUI.

`UC_StagedMediaPipeFaceBackgroundComposite` detects faces in each foreground and adds them as independently placeable layers. The background and face options nodes contain removal, extraction, feathering, and blend settings. `UC_StagedIndividualComposites` provides the same ordinary foreground staging editor but returns one full-background image, placement mask, and box per included foreground without stacking them. `UC_BackgroundRemovalPreserveAlpha` directly returns source-resolution RGBA images and their soft alpha masks; existing RGBA inputs keep their supplied alpha without model execution.

### Staged compositor example

[Workflow JSON](workflows/CompositorExampleWorkflow.json) | [Workflow overview](workflows/CompositorExampleWorkflow.jpg) | [Source assets](workflow_assets/)

<img src="workflows/CompositorExampleWorkflow.jpg" alt="Staged MediaPipe face background compositor workflow" width="1200">

### Resolution and workflow parameters

- `UC_AdjustedResolutionParameters`
- `UC_ResolutionSelectorExtended`
- `UC_VideoResolutionSelector`
- `UC_ImageScaleAndResolutionPicker`
- `UC_SwitchInverseNode`
- `UC_SoftSwitchInverseNode`
- `UC_IntegerRangeRandom`
- `UC_RandInt`
- `UC_StaticInt`
- `UC_StaticFloat`
- `UC_RandIntRange`
- `UC_ColorConvertNode`
- `UC_SeedCluster`
- `UC_FromSeedCluster`
- `UC_ExtractBoundingBox`
- `UC_AdjustBoundingBox`
- `UC_Ideogram4BoundingBoxCrop`
- `UC_HighResolutionTileSplit`
- `UC_HighResolutionTileAccumulator`
- `UC_HighResolutionTilingGuide`

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
- `UC_VLMSysInstrPresetsExperimental`
- `UC_VLMSysInstrLegacyPresets`
- `UC_VLMSysQueryAddPresets`
- `UC_VLMSysQueryRawPresets`
- `UC_VLMSysInstrAdvPresets`
- `UC_VLMSysInstrAdvPresetsExperimental`
- `UC_MiniMaxH3VLMSysInstrPresets`
- `UC_MiniMaxH3VLMSysInstrPresetsExperimental`
- `UC_MiniMaxH3VLMSysInstrAdvPresets`
- `UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental`
- `UC_LegacyPromptPresets`
- `UC_UnifiedPresets`

### Loading, text generation, and text utilities

- `UC_LoadImagePath`
- `UC_LoadImageDirectory`
- `UC_LoadImageWithAlpha`
- `UC_SampleVideoFramesAsImages`
- `UC_ImagesToVideoTimeline`
- `UC_LoraLoaderCLIPOnly`
- `UC_TextGenerate`
- `UC_TextGenerateQwen35SystemPrompt`
- `UC_EmbeddingDetokenizerAnalysis`
- `UC_ImageToVideoPrompt`
- `UC_TagNormalizeCombine`
- `UC_FromList`
- `UC_GetJsonValue`
- `UC_MiniMaxH3Cache`
- `UC_MiniMaxH3Spectrum`
- `UC_MarkdownPreview`
- `UC_BoldFrakturTextStyle`
- `UC_UnBoldFrakturTextStyle`
- `UC_WordJoiner`
- `UC_UnWordJoiner`
- `UC_JSONMinifyRepair`
- `UC_StringUnescape`
- `UC_TextConcatenateAutogrow`
- `UC_TextConcatenateListsAutogrow`
- `UC_Newline`

### Scheduler presets

- `Ideogram4SchedulerPreset`
- `UC_SigmaRescale`
- `UC_DiscardPenultimateSigma`
- `UC_SigmoidOffsetScheduler`
- `UC_PowerShiftScheduler`
- `UC_RadianceShiftScheduler`
- `UC_SigmaCurveFromPointsScheduler`
- `UC_SigmaCurvePchipScheduler`

The migrated schedulers also register `sigmoid_offset`, `power_shift`,
`radiance_shift`, `sigma_curve_from_points`, and `sigma_curve_pchip` for Core
scheduler selectors. The Power Shift scheduler was inspired by
[InverserSquaredScheduler](https://github.com/Clybius/ComfyUI-ClybsChromaNodes/blob/main/clyb_Schedulers.py).

`UC_SigmaRescale` maps an existing schedule to exact start and end sigma
values without changing its shape or number of steps.

The dedicated scheduler nodes do not include Core-style denoise controls or
optional penultimate-sigma controls. Connect `UC_SigmaRescale` after a
scheduler when setting image-to-image noise levels. Connect
`UC_DiscardPenultimateSigma` when the selected sampler requires penultimate
sigma removal. Radiance Shift performs its required compensated removal
internally. Sigmoid Offset retains its model-specific `start_sigma`
adjustment.

### Logic and math

- `UC_LogicIF`
- `UC_LogicAND`
- `UC_LogicOR`
- `UC_LogicNOT`
- `UC_LogicXOR`
- `UC_MathAdd`
- `UC_MathSubtract`
- `UC_MathMultiply`
- `UC_MathDivide`
- `UC_MathPower`
- `UC_MathFloor`
- `UC_MathCeil`
- `UC_MathRound`
- `UC_MathModulo`
- `UC_MathAbs`
- `UC_MathSqrt`
- `UC_MathSin`
- `UC_MathCos`
- `UC_MathTan`
- `UC_MathMin`
- `UC_MathMax`
- `UC_MathClamp`
- `UC_MathNumberConvert`
- `UC_StringToNumber`
- `UC_NumberToString`
- `UC_MathCompare`
- `UC_MathOperation`
- `UC_MathAspectRatio`

These nodes replace the equivalent nodes from ComfyUI-LogicMath, ComfyUI_SigmoidOffsetScheduler,
and ComfyUI_PowerShiftScheduler. Remove the standalone pack before accepting ComfyUI's workflow
replacement prompt.
