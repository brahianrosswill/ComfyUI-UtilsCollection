import re
import math
import os
import gc
import types
import logging
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import comfy
import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io
from comfy.utils import common_upscale
from .helper_functions import get_token_count, get_token_count_scaled
from .encoder_helpers import(
    encode_embedding_scaled_bias,
    is_image_token,
    evaluate_formula,
    evaluate_conditioning_formula,
    evaluate_conditioning_consensus_blend,
    blend_text_vectors,
    find_visual_token_range,
    build_token_to_conditioning_map,
    encode_embedding_classical_scaled_bias,
    strip_contextual_weight_syntax,
    load_vlm_image_tensor,
    krea2_user_content_span,
    krea2_token_ids,
    find_subsequence,
    krea2_attn_forward_weight,
    ImageInputMapping,
    VISION_BLOCK,
    prepare_image_placeholder_prompt,
    extract_and_flatten_images,
    resolve_embedding_output_path,
)

def apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode):
    if not ref_latents:
        return conditioning

    stage = getattr(clip, "cond_stage_model", None)
    stage_names = " ".join(
        type(value).__name__
        for value in (stage, getattr(stage, "clip_model", None), getattr(stage, "clip", None))
        if value is not None
    ).lower()
    if "krea2" in stage_names:
        raise ValueError("Krea2 Core does not consume reference_latents; disable the reference-latent mode.")

    if "parallel" in ref_latent_mode:
        # Keep semantic conditioning and reference-latent conditioning as separate
        # Comfy conditioning entries. Sequence concatenation is not a parallel stream.
        tokens_neutral = clip.tokenize("")
        conditioning_neutral = clip.encode_from_tokens_scheduled(tokens_neutral)
        out = [[tensor, metadata.copy()] for tensor, metadata in conditioning]
        for tensor, metadata in conditioning_neutral:
            neutral_meta = metadata.copy()
            neutral_meta["reference_latents"] = list(ref_latents)
            out.append([tensor, neutral_meta])
        return out
    else:
        # Standard append mode
        return node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)


def multiply_conditioning(conditioning, multiplier):
    if multiplier == 1.0:
        return conditioning
    output = []
    for tensor, metadata in conditioning:
        new_metadata = metadata.copy()
        pooled = new_metadata.get("pooled_output")
        if pooled is not None:
            new_metadata["pooled_output"] = pooled * multiplier
        output.append([tensor * multiplier, new_metadata])
    return output


class UC_AttentionBiasTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_AttentionBiasTextEncode",
            category="advanced/conditioning",
            display_name="CLIP Text Encode with Attention Bias (Experimental)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, text) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        if '<' not in text and '>' not in text and '=' not in text:
            tokens = clip.tokenize(text)
            return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

        bias_pattern = re.compile(r"<([^>]+)=([0-9.-]+)>")
        split_pattern = re.compile(r"(<[^>]+=[0-9.-]+>)")
        segments = split_pattern.split(text)

        clean_text = ""
        biases_to_apply = []

        for segment in segments:
            if not segment:
                continue

            match = bias_pattern.fullmatch(segment)
            if match:
                bias_text, strength_str = match.groups()
                before = clip.tokenize(clean_text)
                key = next(iter(before))
                start_index = len(before[key][0])
                strength = float(strength_str)
                clean_text += bias_text
                after = clip.tokenize(clean_text)
                end_index = len(after[key][0])
                if end_index > start_index:
                    biases_to_apply.append({"start": start_index, "end": end_index, "strength": strength})
            else:
                clean_text += segment

        tokens = clip.tokenize(clean_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if not biases_to_apply:
            return io.NodeOutput(conditioning)

        output = []
        for cond, metadata in conditioning:
            seq_len = cond.shape[1]
            key = next(iter(tokens))
            mapping = build_token_to_conditioning_map(tokens[key][0], cond)
            attn_mask = torch.zeros((1, seq_len, seq_len), dtype=cond.dtype, device=cond.device)
            for bias in biases_to_apply:
                strength = bias["strength"]
                if not math.isfinite(strength) or strength < 0:
                    raise ValueError("Attention weights must be finite and non-negative.")
                value = math.log(max(strength, 1e-6))
                if bias["start"] >= len(mapping):
                    continue
                start = mapping[bias["start"]][0]
                end = mapping[min(bias["end"] - 1, len(mapping) - 1)][1]
                if start < end:
                    # Key-column odds scaling. Row scaling would square the
                    # weighted intersection and is intentionally not applied.
                    attn_mask[:, :, start:end] += value
            new_metadata = metadata.copy()
            existing = new_metadata.get("attention_mask")
            if existing is not None and torch.is_tensor(existing):
                if existing.shape[-1] != seq_len:
                    raise ValueError("Existing attention mask does not match the encoded sequence length.")
                existing = existing.to(device=cond.device)
                if existing.dtype == torch.bool:
                    additive = torch.zeros(existing.shape, device=cond.device, dtype=cond.dtype)
                    additive.masked_fill_(~existing, -torch.finfo(cond.dtype).max)
                else:
                    additive = existing.to(dtype=cond.dtype)
                while additive.ndim < attn_mask.ndim:
                    additive = additive.unsqueeze(-2)
                attn_mask = attn_mask + additive
            new_metadata["attention_mask"] = attn_mask
            new_metadata["attention_mask_img_shape"] = (1, 1)
            output.append([cond, new_metadata])
        return io.NodeOutput(output)

# --- Type Definitions for Modular Configurations ---
TextBlendConfig = io.Custom("TEXT_BLEND_CONFIG")
VisualFusionConfig = io.Custom("VISUAL_FUSION_CONFIG")

class UC_TextConsensusBlendConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_TextConsensusBlendConfig",
            display_name="Text Consensus Blend Configurator",
            category="advanced/conditioning",
            inputs=[
                io.Combo.Input(
                    "blend_preset",
                    options=[
                        "off", "custom", "baseline", "high_clarity", "smooth", "varied_merge", "diverse_concept", "high_diversity_concept",
                        "dsc_baseline", "dsc_high_clarity", "dsc_smooth", "dsc_varied_merge", "dsc_diverse_concept", "dsc_high_diversity_concept"
                    ],
                    default="baseline",
                    tooltip="Preset configuration for Text Consensus-Weighted Blending. Set to 'off' to bypass CWB, or 'custom' to use the manual parameters below."
                ),
                io.Combo.Input(
                    "blend_method",
                    options=["linear", "consensus"],
                    default="consensus",
                    tooltip="Active only in 'custom' preset. 'consensus' aligns prompts and filters noise; 'linear' averages them."
                ),
                io.Combo.Input(
                    "consensus_type",
                    options=["mean", "median"],
                    default="median",
                    tooltip="Active only in 'custom' preset. 'median' rejects up to 50% outlying noise; 'mean' is smooth averaging."
                ),
                io.Combo.Input(
                    "alignment_method",
                    options=["index", "similarity"],
                    default="similarity",
                    tooltip="Active only in 'custom' preset. 'similarity' aligns shifted prompt concepts; 'index' aligns them sequentially."
                ),
                io.Float.Input("alignment_threshold", default=0.4, min=0.0, max=1.0, step=0.01, tooltip="Active only in similarity alignment. Minimum similarity to match words."),
                io.Float.Input("similarity_threshold", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Prunes passing words if similarity to consensus falls below this."),
                io.Float.Input("power_alpha", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="Soft-masking exponent. Higher values penalize outliers (e.g. 2.0)."),
                io.Float.Input("diversity_beta", default=0.0, min=0.0, max=10.0, step=0.1, tooltip="Diversity exponent. Dampens hyper-frequent details to boost variety (e.g. 1.5)."),
                io.Boolean.Input("rescale_norm", default=True, tooltip="Norm Rescaling. Keeps activation energy high to prevent washed-out colors."),
                io.Float.Input("global_scale", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Global scale multiplier applied to the blended outputs."),
                io.Boolean.Input("dynamic_similarity_contrast", default=False, tooltip="Stretches similarities to soft [0.7, 1.0] band to boost contrast."),
                io.Boolean.Input("soft_comfort_bandpass", default=False, tooltip="Softens the diversity bandpass ceiling to prevent clipping.")
            ],
            outputs=[
                TextBlendConfig.Output("text_blend_config")
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        blend_preset: str,
        blend_method: str,
        consensus_type: str,
        alignment_method: str,
        alignment_threshold: float,
        similarity_threshold: float,
        power_alpha: float,
        diversity_beta: float,
        rescale_norm: bool,
        global_scale: float,
        dynamic_similarity_contrast: bool = False,
        soft_comfort_bandpass: bool = False
    ) -> io.NodeOutput:
        config = {
            "blend_preset": blend_preset,
            "blend_method": blend_method,
            "consensus_type": consensus_type,
            "alignment_method": alignment_method,
            "alignment_threshold": alignment_threshold,
            "similarity_threshold": similarity_threshold,
            "power_alpha": power_alpha,
            "diversity_beta": diversity_beta,
            "rescale_norm": rescale_norm,
            "global_scale": global_scale,
            "dynamic_similarity_contrast": dynamic_similarity_contrast,
            "soft_comfort_bandpass": soft_comfort_bandpass
        }
        return io.NodeOutput(config)


class UC_VisualFusionConfig(io.ComfyNode):
    """
    Configuration node for visual component fusion.
    Specifies methods for blending or spatially interleaving isolated visual token vectors,
    and provides controls to save dynamically blended visual embeddings directly to disk.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_VisualFusionConfig",
            display_name="Visual Component Fusion Configurator",
            category="advanced/conditioning",
            inputs=[
                io.Combo.Input(
                    "visual_fusion_method",
                    options=["off", "linear", "spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"],
                    default="spatial-checkerboard",
                    tooltip="Method to combine isolated visual-token vectors. Spatial methods select source vectors according to a reproducible token-grid pattern; generation quality is model and prompt dependent."
                ),
                io.Int.Input("visual_block_size", default=2, min=1, max=8, step=1, tooltip="Active for spatial-block-interleave. Size of the spatial token patches to group and switch together."),
                io.Float.Input("dither_ratio", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Active for spatial-dither-random. Probability of selecting the first image. Remaining images are selected with a checkerboard pattern."),
                io.Boolean.Input("save_blended_embeds", default=False, tooltip="Enable to save the blended visual tokens as a standalone .safetensors embedding."),
                io.String.Input("save_path", default="blended_visual_embeds.safetensors", tooltip="Target filename/path under models/embeddings to save the .safetensors file."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, tooltip="Seed for the spatial-dither-random pattern.")
            ],
            outputs=[
                VisualFusionConfig.Output("visual_fusion_config")
            ]
        )

    @classmethod
    def execute(
        cls,
        visual_fusion_method: str,
        visual_block_size: int,
        dither_ratio: float,
        save_blended_embeds: bool = False,
        save_path: str = "blended_visual_embeds.safetensors",
        seed: int = 0,
    ) -> io.NodeOutput:
        config = {
            "visual_fusion_method": visual_fusion_method,
            "visual_block_size": visual_block_size,
            "dither_ratio": dither_ratio,
            "seed": seed,
            "save_blended_embeds": save_blended_embeds,
            "save_path": save_path
        }
        return io.NodeOutput(config)

class UC_ConditioningConsensusBlend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Conditioning.Input("conditioning", optional=True),
            prefix="conditioning_"
        )
        return io.Schema(
            node_id="UC_ConditioningConsensusBlend",
            display_name="Conditioning Consensus Blender (Post-Encoder)",
            category="advanced/conditioning",
            inputs=[
                io.Autogrow.Input("conditioning_inputs", template=autogrow_template),
                TextBlendConfig.Input("text_blend_config", optional=True, tooltip="Optional configuration from UC_TextConsensusBlendConfig. Defaults to baseline CWB if disconnected.")
            ],
            outputs=[
                io.Conditioning.Output("conditioning")
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, conditioning_inputs: io.Autogrow.Type, text_blend_config: dict = None) -> io.NodeOutput:
        """
        Blends stock or custom ComfyUI conditioning outputs post-encoder using CWB math.
        """
        if not conditioning_inputs:
            raise ValueError("At least one conditioning input must be connected to UC_ConditioningConsensusBlend.")

        active_conds = []
        if conditioning_inputs is not None:
            for k in sorted(conditioning_inputs.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)):
                v = conditioning_inputs[k]
                if v is not None:
                    active_conds.append(v)

        if not active_conds:
            raise ValueError("All connected conditioning inputs to UC_ConditioningConsensusBlend are empty or None.")

        if len(active_conds) == 1:
            return io.NodeOutput(active_conds[0])

        if text_blend_config is None:
            text_blend_config = {"blend_preset": "baseline"}

        if text_blend_config.get("blend_preset") == "off":
            return io.NodeOutput(active_conds[0])

        schedule_lengths = {len(conditioning) for conditioning in active_conds}
        if len(schedule_lengths) != 1:
            raise ValueError("All conditioning inputs must have the same number of scheduled entries.")

        def compatible_metadata(metadata_items):
            layout_keys = {"attention_mask", "attention_mask_img_shape", "embeds_info"}
            result = {}
            common_keys = set.intersection(*(set(item) for item in metadata_items)) - layout_keys - {"pooled_output"}
            for key in common_keys:
                values = [item[key] for item in metadata_items]
                first = values[0]
                if torch.is_tensor(first):
                    if all(torch.is_tensor(value) and value.shape == first.shape and torch.equal(value, first) for value in values[1:]):
                        result[key] = first
                elif all(value is first for value in values[1:]):
                    result[key] = first
                elif isinstance(first, (str, int, float, bool, type(None))) and all(value == first for value in values[1:]):
                    result[key] = first
            return result

        device = comfy.model_management.get_torch_device()
        blended_conditioning = []
        for schedule_index in range(next(iter(schedule_lengths))):
            entries = [conditioning[schedule_index] for conditioning in active_conds]
            dtype = entries[0][0].dtype
            sequence_tensors = {}
            pooled_tensors = {}
            for index, (tensor, metadata) in enumerate(entries):
                key = chr(97 + index)
                sequence_tensors[key] = comfy.model_management.cast_to_device(tensor, device, dtype)
                pooled = metadata.get("pooled_output") if metadata else None
                pooled_tensors[key] = (
                    comfy.model_management.cast_to_device(pooled, device, dtype) if pooled is not None else None
                )
            C_blended, P_blended = blend_text_vectors(
                sequence_tensors,
                text_blend_config,
                pooled_tensors=pooled_tensors,
                device=str(device),
            )
            metadata = compatible_metadata([entry[1] for entry in entries])
            if P_blended is not None:
                metadata["pooled_output"] = P_blended
            blended_conditioning.append([C_blended, metadata])
        return io.NodeOutput(blended_conditioning)

class UC_ScaledBiasTextEncodeFlux2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeFlux2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 dev System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = r"[SYSTEM_PROMPT]"
            template_suffix = r"[/SYSTEM_PROMPT][INST]{}[/INST]"
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeKleinSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeKleinSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 Klein System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )
        else:
            llama_template = (
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, skip_template=True)
        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeLtxv2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeLtxv2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with LTXV 2 System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, image=None) -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if image is not None:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n\n<image_soft_token>{}<end_of_turn>\n\n<start_of_turn>model\n"
            )
        elif len(system_prompt) > 0:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            llama_template = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, image=image)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off" and image is not None:
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if vae_resolution == "Original":
                width_vae = round(samples.shape[3] / 8.0) * 8
                height_vae = round(samples.shape[2] / 8.0) * 8
                s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
            else:
                vae_size = VAE_RESOLUTIONS[vae_resolution]
                total_vae = vae_size * vae_size
                scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeZITSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeZITSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = "<|im_start|>system\n"
            template_suffix = (
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeZImageThinkPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeZImageThinkPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image Thinking Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("thinking", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, thinking=None) -> io.NodeOutput:
        if len(thinking) > 0:
            template_prefix = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            template_suffix = "\n</think>\n\n"
            llama_template = f"{template_prefix}{thinking}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "z-image"],
                    default="flux2dev",
                    tooltip="Select the model type to use the correct template format.",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="(Klein only) Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        skip_template = False
        if model_type == "klein" and len(thinking_content) > 0:
            # Klein with custom thinking content
            if len(system_prompt) > 0:
                llama_template = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n" +
                    f"<|im_start|>user\n{{}}<|im_end|>\n" +
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            else:
                llama_template = (
                    "<|im_start|>user\n{}<|im_end|>\n" +
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            skip_template = True
        elif len(system_prompt) > 0:
            template = SYSTEM_PROMPT_TEMPLATES.get(model_type, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
            llama_template = f"{template['prefix']}{system_prompt}{template['suffix']}"
            skip_template = (model_type == "klein")
        else:
            llama_template = None

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, skip_template=skip_template)
        return io.NodeOutput(conditioning)


class UC_TextEncodeFlux2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeFlux2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 dev System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = r"[SYSTEM_PROMPT]"
            template_suffix = r"[/SYSTEM_PROMPT][INST]{}[/INST]"
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeKleinSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeKleinSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 Klein System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )
        else:
            llama_template = (
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )

        tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeKrea2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeKrea2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Krea2 System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            llama_template = (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeSystemEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeSystemEditPlus",
            display_name="TextEncodeSystemEditPlus",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path). 'Fast' = 1024x1024, 'Balanced' = 1280x1280, 'Detailed' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, vae_resolution, ref_latent_mode="off", vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        image_prompt = ""

        VLM_RESOLUTIONS = {
            "Fast (384)": 384,
            "Balanced (512)": 512,
            "Detailed (768)": 768,
            "Large (1024)": 1024,
            "X-Large (1280)": 1280,
            "XX-Large (1536)": 1536
        }

        VAE_RESOLUTIONS = {
            "Ultra (512)": 512,
            "Turbo (768)": 768,
            "Fast (1024)": 1024,
            "Balanced (1280)": 1280,
            "Detailed (1536)": 1536
        }

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)

                # 1. Semantic Path Scaling (VLM)
                if vlm_resolution == "Original":
                    images_vl.append(image)
                else:
                    vlm_size = VLM_RESOLUTIONS[vlm_resolution]
                    total_vlm = vlm_size * vlm_size
                    scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                    width_vlm = round(samples.shape[3] * scale_by_vlm)
                    height_vlm = round(samples.shape[2] * scale_by_vlm)

                    s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                    images_vl.append(s_vlm.movedim(1, -1))

                # 2. Structural Path Scaling (VAE)
                if vae is not None and ref_latent_mode != "off":
                    if "multi" in ref_latent_mode or len(ref_latents) == 0:
                        if vae_resolution == "Original":
                            width_vae = round(samples.shape[3] / 8.0) * 8
                            height_vae = round(samples.shape[2] / 8.0) * 8
                            s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                            ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                        else:
                            vae_size = VAE_RESOLUTIONS[vae_resolution]
                            total_vae = vae_size * vae_size
                            scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                            width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                            height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                            s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                            ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # Construct the complete template string via safe concatenation to prevent formatting errors
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + image_prompt + prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + image_prompt + prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeSystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeSystemEditPlusAdvanced",
            display_name="TextEncodeSystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None) -> io.NodeOutput:
        # Collect, extract, and parse all autogrow keys (including batched images)
        raw_images, flat_images, is_zero_indexed = extract_and_flatten_images(image_inputs)

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process in sorted order of raw_images keys
            for num in sorted(raw_images.keys()):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                image = raw_images[num]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    if vae_resolution == "Original":
                        width_vae = round(samples.shape[3] / 8.0) * 8
                        height_vae = round(samples.shape[2] / 8.0) * 8
                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                    else:
                        vae_size = VAE_RESOLUTIONS[vae_resolution]
                        total_vae = vae_size * vae_size
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeKrea2SystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeKrea2SystemEditPlusAdvanced",
            display_name="TextEncodeKrea2SystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Main text prompt. Supports visual math blending: |formula| to blend image inputs at pixel-tensor level before encoding. "
                        "Example: |((image_input_1 * 1.075) + (image_input_2 * 1.025)) / 1.5| to blend styles/concepts. "
                        "Supported math operations: +, -, *, /, clamp, min, max, abs, on variables image_input_1 to image_input_16."
                    ),
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None) -> io.NodeOutput:
        # Collect, extract, and parse all autogrow keys (including batched images)
        raw_images, flat_images, is_zero_indexed = extract_and_flatten_images(image_inputs)

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process in sorted order of raw_images keys
            for num in sorted(raw_images.keys()):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                image = raw_images[num]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    if vae_resolution == "Original":
                        width_vae = round(samples.shape[3] / 8.0) * 8
                        height_vae = round(samples.shape[2] / 8.0) * 8
                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                    else:
                        vae_size = VAE_RESOLUTIONS[vae_resolution]
                        total_vae = vae_size * vae_size
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeEditPlusAdvanced",
            display_name="TextEncodeEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Main text prompt. Supports visual math blending: |formula| to blend image inputs at pixel-tensor level before encoding. "
                        "Example: |((image_input_1 * 1.075) + (image_input_2 * 1.025)) / 1.5| to blend styles/concepts. "
                        "Supported math operations: +, -, *, /, clamp, min, max, abs, on variables image_input_1 to image_input_16."
                    ),
                ),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, image_inputs: io.Autogrow.Type, vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None) -> io.NodeOutput:
        # Collect, extract, and parse all autogrow keys (including batched images)
        raw_images, flat_images, is_zero_indexed = extract_and_flatten_images(image_inputs)

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Pass standard tokens to tokenize (with images mapped to tags) and encode
        tokens = clip.tokenize(modified_prompt, images=images_vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process in sorted order of raw_images keys
            for num in sorted(raw_images.keys()):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                image = raw_images[num]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    if vae_resolution == "Original":
                        width_vae = round(samples.shape[3] / 8.0) * 8
                        height_vae = round(samples.shape[2] / 8.0) * 8
                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                    else:
                        vae_size = VAE_RESOLUTIONS[vae_resolution]
                        total_vae = vae_size * vae_size
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeGemmaSystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeGemmaSystemEditPlusAdvanced",
            display_name="TextEncodeGemmaSystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None) -> io.NodeOutput:
        # Collect, extract, and parse all autogrow keys (including batched images)
        raw_images, flat_images, is_zero_indexed = extract_and_flatten_images(image_inputs)

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)
        has_keywords = bool(pattern.search(prompt))

        images_vl_raw = []

        if has_keywords:
            # Replace keywords dynamically and build images_vl_raw in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = num - 1 if is_zero_indexed else num
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    images_vl_raw.append(img)
                    return "<img><image_soft_token><end_of_image>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                img = raw_images[num]
                images_vl_raw.append(img)
                display_num = num + 1 if is_zero_indexed else num
                image_prompt += f"<img><image_soft_token><end_of_image>"

            modified_prompt = image_prompt + prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n" + modified_prompt + "<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            full_prompt = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n" +
                "<start_of_turn>user\n" + modified_prompt + "<end_of_turn>\n<start_of_turn>model\n"
            )

        # 1. First tokenize the text without passing images, getting raw 262144 token IDs
        tokens = clip.tokenize(full_prompt, skip_template=True)

        # 2. Helper to process image for VLM
        def process_vlm_image(image, res):
            if image is None:
                return None
            if res == "Original":
                return image[:, :, :, :3]
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            vlm_size = VLM_RESOLUTIONS[res]
            total_vlm = vlm_size * vlm_size
            scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
            width_vlm = round(samples.shape[3] * scale_by_vlm)
            height_vlm = round(samples.shape[2] * scale_by_vlm)

            s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
            return s_vlm.movedim(1, -1)[:, :, :, :3]

        # 3. Process the images and manually inject them sequentially into the 262144 tokens
        if len(images_vl_raw) > 0:
            processed_images = [process_vlm_image(img, vlm_resolution) for img in images_vl_raw]

            # Loop over all tokenizer sections (e.g. 'gemma3_12b')
            for key, val in tokens.items():
                if isinstance(val, list):
                    embed_count = 0
                    for r in val:
                        if isinstance(r, list):
                            for i, token in enumerate(r):
                                if isinstance(token, tuple) and len(token) > 0:
                                    if token[0] == 262144 and embed_count < len(processed_images):
                                        # Replace the token ID (index 0 of the tuple) with the visual payload dict
                                        r[i] = ({"type": "image", "data": processed_images[embed_count]},) + token[1:]
                                        embed_count += 1

        # 4. Encode from the modified tokens dict
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process in order of images_vl_raw
            for i, image in enumerate(images_vl_raw):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                if image is not None:
                    samples = image.movedim(-1, 1)
                    if vae_resolution == "Original":
                        width_vae = round(samples.shape[3] / 8.0) * 8
                        height_vae = round(samples.shape[2] / 8.0) * 8
                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                    else:
                        vae_size = VAE_RESOLUTIONS[vae_resolution]
                        total_vae = vae_size * vae_size
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class UC_TextEncodeLtxv2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeLtxv2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with LTXV 2 System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, image=None) -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if image is not None:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n\n<image_soft_token>{}<end_of_turn>\n\n<start_of_turn>model\n"
            )
        elif len(system_prompt) > 0:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            llama_template = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )

        if image is not None:
            tokens = clip.tokenize(prompt, llama_template=llama_template, image=image)
        else:
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off" and image is not None:
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if vae_resolution == "Original":
                width_vae = round(samples.shape[3] / 8.0) * 8
                height_vae = round(samples.shape[2] / 8.0) * 8
                s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
            else:
                vae_size = VAE_RESOLUTIONS[vae_resolution]
                total_vae = vae_size * vae_size
                scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)

        return io.NodeOutput(conditioning)


class UC_TextEncodeZITSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeZITSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = "<|im_start|>system\n"
            template_suffix = (
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeZImageThinkPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeZImageThinkPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image Thinking Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("thinking", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, thinking=None) -> io.NodeOutput:
        if len(thinking) > 0:
            template_prefix = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            template_suffix = "\n</think>\n\n"
            llama_template = f"{template_prefix}{thinking}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


# Template definitions for unified node
SYSTEM_PROMPT_TEMPLATES = {
    "flux2dev": {
        "prefix": r"[SYSTEM_PROMPT]",
        "suffix": r"[/SYSTEM_PROMPT][INST]{}[/INST]",
    },
    "klein": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    "z-image": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    },
    "krea2": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    },
}


def system_prompt_template(model_type, system_prompt, thinking_content=""):
    if model_type == "klein" and thinking_content:
        if system_prompt:
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
            ), True
        return (
            "<|im_start|>user\n{}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
        ), True
    if model_type == "z-image-thinking" and thinking_content:
        system_block = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
        return (
            system_block
            + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            + f"{thinking_content}\n</think>\n\n"
        ), False
    if system_prompt:
        template_key = "z-image" if model_type == "z-image-thinking" else model_type
        template = SYSTEM_PROMPT_TEMPLATES.get(template_key, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
        return f"{template['prefix']}{system_prompt}{template['suffix']}", model_type == "klein"
    return None, False


class UC_TextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeSystemPrompt",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "krea2", "z-image", "z-image-thinking"],
                    default="flux2dev",
                    tooltip="Select the model type to use the correct template format.",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content for Klein or the z-image-thinking profile. Leave empty for the model default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        llama_template, skip_template = system_prompt_template(model_type, system_prompt, thinking_content)
        if llama_template is None:
            tokens = clip.tokenize(prompt)
        else:
            tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=skip_template)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_WeightedTextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_WeightedTextEncodeSystemPrompt",
            display_name="Weighted System Prompt Text Encode",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "krea2", "z-image", "z-image-thinking"],
                    default="flux2dev",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input("thinking_content", multiline=True, dynamic_prompts=True, default=""),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="", multiplier=1.0):
        llama_template, skip_template = system_prompt_template(model_type, system_prompt, thinking_content)
        conditioning = encode_embedding_classical_scaled_bias(
            clip,
            prompt,
            llama_template=llama_template,
            skip_template=skip_template,
        )
        return io.NodeOutput(multiply_conditioning(conditioning, multiplier))




class TextEncodeKrea2SystemEditScaledAdv(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeKrea2SystemEditScaledAdv",
            display_name="Krea2 System Prompt Scaled Encoder (Advanced)",
            category="advanced/conditioning",
            inputs=[
                # --- Primary Inputs ---
                io.Clip.Input("clip", tooltip="CLIP/T5 dual text encoder reference."),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main prompt. With fusion off, image_input_N places active image N inline. With fusion on, use image_input_fusion (image_input_1 is accepted as an alias).",
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default="", tooltip="System prompt injected prior to user description."),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Multimodal images. Maps active inputs sequentially to variables (a, b, c, ...)."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),

                # --- Modular Configurations ---
                VisualFusionConfig.Input("visual_fusion_config", optional=True, tooltip="Optional spatial visual fusion configuration from UC_VisualFusionConfig. Blends isolated visual blocks without coordinate blur."),

                # --- Fallback Controls ---
                io.String.Input(
                    "formula",
                    default="a",
                    multiline=False,
                    tooltip="Mathematical formula used with fusion off when no numbered inline placeholders are present. Use a, b, c, d... for active image passes.",
                ),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions. Active ONLY if visual_fusion_config is disconnected or set to 'off'.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, visual_fusion_config: dict = None, formula: str = "a", padding_method: str = "zero-pad", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, multiplier: float = 1.0) -> io.NodeOutput:
        # Collect, extract, and parse all active (non-null) connected images sequentially (including batched images)
        _, active_images, _ = extract_and_flatten_images(image_inputs)

        if not active_images:
            # Fallback if no images are connected: encode prompt as plain text
            logging.warning("AdvancedVisualConditioning: no images are connected; encoding the prompt as text only.")
            clean_prompt, _ = prepare_image_placeholder_prompt(
                prompt,
                image_count=0,
                fusion_active=False,
                context="AdvancedVisualConditioning",
            )
            conditioning = multiply_conditioning(encode_embedding_classical_scaled_bias(clip, clean_prompt), multiplier)
            return io.NodeOutput(conditioning)

        # Map active images sequentially to letter variables (a, b, c, ...) and encode each pass
        sequence_tensors = {}
        pooled_tensors = {}
        visual_ranges = {}
        tokens_dict = {}
        reference_cond_dict = None

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "off"}
        visual_method = visual_fusion_config.get("visual_fusion_method", "off")

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        def format_krea_prompt(user_prompt):
            if system_prompt:
                return (
                    "<|im_start|>user\n<|im_end|>\n"
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            return (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        processed_active_images = [process_vlm_image(image, vlm_resolution) for image in active_images]
        prepared_prompt, inline_numbers = prepare_image_placeholder_prompt(
            prompt,
            image_count=len(processed_active_images),
            fusion_active=visual_method != "off",
            context="AdvancedVisualConditioning",
        )

        inline_mode = False
        if visual_method == "off" and inline_numbers:
            inline_images = [processed_active_images[number - 1] for number in inline_numbers]
            inline_prompt = format_krea_prompt(prepared_prompt)
            if strip_contextual_weight_syntax(inline_prompt) != inline_prompt:
                logging.warning(
                    "AdvancedVisualConditioning: custom contextual vector scaling is disabled for native multi-image inline encoding."
                )
            try:
                inline_tokens = clip.tokenize(inline_prompt, images=inline_images, skip_template=True)
                inline_cond = clip.encode_from_tokens_scheduled(inline_tokens)
                if len(inline_cond) != 1:
                    raise ValueError("Inline image encoding requires a single conditioning schedule entry.")
            except (TypeError, ValueError) as exc:
                logging.warning(
                    "AdvancedVisualConditioning: inline placeholder encoding failed (%s); falling back to per-image formula encoding.",
                    exc,
                )
                prepared_prompt, _ = prepare_image_placeholder_prompt(
                    prompt,
                    image_count=0,
                    fusion_active=False,
                    context="AdvancedVisualConditioning fallback",
                )
            except Exception:
                logging.exception("AdvancedVisualConditioning: inline image encoding failed and cannot be safely recovered.")
                raise
            else:
                inline_mode = True
                sequence_tensors["a"] = inline_cond[0][0]
                pooled_output = inline_cond[0][1].get("pooled_output")
                if pooled_output is not None:
                    pooled_tensors["a"] = pooled_output
                reference_cond_dict = inline_cond[0][1]
                if formula.strip() != "a":
                    logging.warning(
                        "AdvancedVisualConditioning: numbered inline placeholders encode one native multimodal sequence; formula '%s' was ignored.",
                        formula,
                    )
                formula = "a"

        multipass_images = [] if inline_mode else processed_active_images

        for idx, processed_img in enumerate(multipass_images):
            letter = chr(97 + idx)  # 0 -> 'a', 1 -> 'b', 2 -> 'c', ...

            # Ensure prompt has image pad tokens so tokenizer knows where to inject the image
            modified_prompt = prepared_prompt
            if not any(tag in modified_prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                modified_prompt = VISION_BLOCK + modified_prompt
            full_prompt = format_krea_prompt(modified_prompt)

            # Encode individual sequence pass
            cond_X = encode_embedding_classical_scaled_bias(clip, full_prompt, images=[processed_img], skip_template=True)
            if len(cond_X) != 1:
                raise ValueError("Advanced visual fusion requires a single conditioning schedule entry.")
            C_X = cond_X[0][0]
            P_X = cond_X[0][1].get("pooled_output", None)

            sequence_tensors[letter] = C_X
            if P_X is not None:
                pooled_tensors[letter] = P_X

            if visual_method != "off":
                try:
                    tokens = clip.tokenize(full_prompt, images=[processed_img], skip_template=True)
                    tokens_dict[letter] = tokens
                    vis_start, vis_end = find_visual_token_range(tokens, C_X)
                    visual_ranges[letter] = (vis_start, vis_end)
                except Exception as e:
                    raise ValueError(f"Could not locate the visual token range for image {idx + 1}: {e}") from e

            if reference_cond_dict is None:
                reference_cond_dict = cond_X[0][1]

        # Evaluate mathematical formula or consensus on sequence and pooled tensors
        fusion_mask_cache = {}
        if visual_method != "off":
            device = comfy.model_management.get_torch_device()

            key_name = "qwen3vl_8b"
            if "tokens" in locals() and tokens:
                key_name = next(iter(tokens.keys()))

            C_blended, P_blended = evaluate_conditioning_consensus_blend(
                sequence_tensors, pooled_tensors, visual_fusion_config=visual_fusion_config, device=device, visual_ranges=visual_ranges, embedding_key=key_name, clip=clip, tokens_dict=tokens_dict, mask_cache=fusion_mask_cache
            )
        else:
            C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

        # Build final conditioning dictionary
        final_cond_dict = reference_cond_dict.copy()
        attention_mask = final_cond_dict.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.shape[-1] != C_blended.shape[1]:
            final_cond_dict.pop("attention_mask", None)
            final_cond_dict.pop("attention_mask_img_shape", None)
        if P_blended is not None:
            final_cond_dict["pooled_output"] = P_blended

        if multiplier != 1.0:
            C_blended *= multiplier
            if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                final_cond_dict["pooled_output"] *= multiplier

        conditioning = [[C_blended, final_cond_dict]]

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process sequentially from active_images
            for i, image in enumerate(active_images):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                if image is not None:
                    samples = image.movedim(-1, 1)
                    if vae_resolution == "Original":
                        width_vae = round(samples.shape[3] / 8.0) * 8
                        height_vae = round(samples.shape[2] / 8.0) * 8
                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                    else:
                        vae_size = VAE_RESOLUTIONS[vae_resolution]
                        total_vae = vae_size * vae_size
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeEditScaledAdv(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeEditScaledAdv",
            display_name="Text Scaled Encoder (Advanced)",
            category="advanced/conditioning",
            inputs=[
                # --- Primary Inputs ---
                io.Clip.Input("clip", tooltip="CLIP/T5 dual text encoder reference."),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main user text prompt. Supports classical weight syntax: (prompt:weight), e.g. (sunset:1.2).",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Multimodal images. Maps active inputs sequentially to variables (a, b, c, ...)."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),

                # --- Modular Configurations ---
                VisualFusionConfig.Input("visual_fusion_config", optional=True, tooltip="Optional spatial visual fusion configuration from UC_VisualFusionConfig. Blends isolated visual blocks without coordinate blur."),

                # --- Fallback Controls ---
                io.String.Input(
                    "formula",
                    default="a",
                    multiline=False,
                    tooltip="Mathematical formula to blend conditioning outputs. Active ONLY if visual_fusion_config is disconnected or set to 'off'. Use variables a, b, c, d... to reference active connected image inputs.",
                ),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions. Active ONLY if visual_fusion_config is disconnected or set to 'off'.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, image_inputs: io.Autogrow.Type, visual_fusion_config: dict = None, formula: str = "a", padding_method: str = "zero-pad", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, multiplier: float = 1.0) -> io.NodeOutput:
        # Collect, extract, and parse all active (non-null) connected images sequentially (including batched images)
        _, active_images, _ = extract_and_flatten_images(image_inputs)

        if not active_images:
            # Fallback if no images are connected: encode prompt as plain text
            conditioning = multiply_conditioning(encode_embedding_classical_scaled_bias(clip, prompt), multiplier)
            return io.NodeOutput(conditioning)

        # Map active images sequentially to letter variables (a, b, c, ...) and encode each pass
        sequence_tensors = {}
        pooled_tensors = {}
        visual_ranges = {}
        tokens_dict = {}
        reference_cond_dict = None

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "off"}
        visual_method = visual_fusion_config.get("visual_fusion_method", "off")

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        for idx, img in enumerate(active_images):
            letter = chr(97 + idx)  # 0 -> 'a', 1 -> 'b', 2 -> 'c', ...
            processed_img = process_vlm_image(img, vlm_resolution)

            # Ensure prompt has image pad tokens so tokenizer knows where to inject the image
            modified_prompt = prompt
            if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            # Encode individual sequence pass
            cond_X = encode_embedding_classical_scaled_bias(clip, modified_prompt, images=[processed_img])
            if len(cond_X) != 1:
                raise ValueError("Advanced visual fusion requires a single conditioning schedule entry.")
            C_X = cond_X[0][0]
            P_X = cond_X[0][1].get("pooled_output", None)

            sequence_tensors[letter] = C_X
            if P_X is not None:
                pooled_tensors[letter] = P_X

            if visual_method != "off":
                try:
                    tokens = clip.tokenize(modified_prompt, images=[processed_img], skip_template=True)
                    tokens_dict[letter] = tokens
                    vis_start, vis_end = find_visual_token_range(tokens, C_X)
                    visual_ranges[letter] = (vis_start, vis_end)
                except Exception as e:
                    raise ValueError(f"Could not locate the visual token range for image {idx + 1}: {e}") from e

            if reference_cond_dict is None:
                reference_cond_dict = cond_X[0][1]

        # Evaluate mathematical formula or consensus on sequence and pooled tensors
        fusion_mask_cache = {}
        if visual_method != "off":
            device = comfy.model_management.get_torch_device()

            key_name = "qwen3vl_8b"
            if "tokens" in locals() and tokens:
                key_name = next(iter(tokens.keys()))

            C_blended, P_blended = evaluate_conditioning_consensus_blend(
                sequence_tensors, pooled_tensors, visual_fusion_config=visual_fusion_config, device=device, visual_ranges=visual_ranges, embedding_key=key_name, clip=clip, tokens_dict=tokens_dict, mask_cache=fusion_mask_cache
            )
        else:
            C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

        # Build final conditioning dictionary
        final_cond_dict = reference_cond_dict.copy()
        attention_mask = final_cond_dict.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.shape[-1] != C_blended.shape[1]:
            final_cond_dict.pop("attention_mask", None)
            final_cond_dict.pop("attention_mask_img_shape", None)
        if P_blended is not None:
            final_cond_dict["pooled_output"] = P_blended

        if multiplier != 1.0:
            C_blended *= multiplier
            if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                final_cond_dict["pooled_output"] *= multiplier

        conditioning = [[C_blended, final_cond_dict]]

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process sequentially from active_images
            for i, image in enumerate(active_images):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                if image is not None:
                    samples = image.movedim(-1, 1)
                    if vae_resolution == "Original":
                        width_vae = round(samples.shape[3] / 8.0) * 8
                        height_vae = round(samples.shape[2] / 8.0) * 8
                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                    else:
                        vae_size = VAE_RESOLUTIONS[vae_resolution]
                        total_vae = vae_size * vae_size
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class UC_Krea2InputEmbeds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_Krea2InputEmbeds",
            display_name="Krea 2 Input Embeddings",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Input text prompt. Important: skips any template wrapping."),
                io.String.Input("image_paths", multiline=True, default="", placeholder="C:/paths/to/image1.png\nC:/paths/to/image2.png", tooltip="Line-separated list of paths to image files. Must map 1-to-1 with file_names."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.String.Input("file_names", multiline=True, default="", placeholder="bulbasaur\nivysaur", tooltip="Line-separated list of file names to save as (without .safetensors). Can include nested subfolders. Must map 1-to-1 with image_paths."),
                io.Boolean.Input(
                    "slice_visual_tokens",
                    default=False,
                    tooltip="If True, removes the first validated visual-token span. If False, preserves the full interleaved sequence.",
                ),
            ],
            outputs=[
                io.AnyType.Output("state_dict", tooltip="Dictionary structure: {'qwen3vl_4b': tensor_2d} of shape [num_tokens, 2560]"),
                io.AnyType.Output("tensor_2d", tooltip="Raw PyTorch 2D tensor of shape [num_tokens, 2560]"),
            ]
        )

    @classmethod
    def execute(cls, clip, prompt, image_paths, vlm_resolution, file_names, slice_visual_tokens=False) -> io.NodeOutput:
        # 1. Parse image paths and file names
        img_paths_list = [p.strip() for p in image_paths.split("\n") if p.strip()]
        file_names_list = [n.strip() for n in file_names.split("\n") if n.strip()]

        if not img_paths_list and not file_names_list:
            raise ValueError("Both image_paths and file_names are empty.")

        # If only text prompt encoding is desired (no images)
        if not img_paths_list:
            if not file_names_list:
                raise ValueError("No file_names specified to save the text embedding.")
            img_paths_list = [None] * len(file_names_list)
        elif not file_names_list:
            raise ValueError("No file_names specified for the provided image paths.")

        if len(img_paths_list) != len(file_names_list):
            raise ValueError(f"Count mismatch: Got {len(img_paths_list)} image paths and {len(file_names_list)} file names.")

        # 2. Pre-Execution Path Validation
        for path in img_paths_list:
            if path is not None:
                normalized_path = path.strip().replace('\\', '/')
                normalized_path = os.path.normpath(normalized_path)
                if not os.path.isabs(normalized_path):
                    normalized_path = os.path.abspath(normalized_path)
                if not os.path.isfile(normalized_path):
                    raise FileNotFoundError(
                        f"Validation aborted: Image file does not exist: '{path}' (resolved to: '{normalized_path}'). "
                        "No processing has started, ensuring safe memory state."
                    )

        # 3. Call clip.load_model() once to register the model as active for comfy-aimdo
        clip.load_model()
        cond_stage = clip.cond_stage_model
        clip_model = None

        if hasattr(cond_stage, "clip") and isinstance(cond_stage.clip, str) and hasattr(cond_stage, cond_stage.clip):
            clip_model = getattr(cond_stage, cond_stage.clip)
        elif hasattr(cond_stage, "clip_model"):
            clip_model = cond_stage.clip_model
        elif hasattr(cond_stage, "clip_d"):
            clip_model = cond_stage.clip_d
        else:
            clip_model = cond_stage

        if clip_model is None or not hasattr(clip_model, "process_tokens"):
            raise AttributeError("Could not locate underlying model wrapper with 'process_tokens' method in cond_stage_model.")


        # Locate ComfyUI embeddings directory
        try:
            embed_paths = folder_paths.get_folder_paths("embeddings")
            if embed_paths:
                embeddings_dir = embed_paths[0]
            else:
                embeddings_dir = os.path.join(os.path.dirname(folder_paths.__file__), "models", "embeddings")
        except Exception:
            embeddings_dir = "models/embeddings"

        os.makedirs(embeddings_dir, exist_ok=True)

        last_state_dict = None
        last_tensor_2d = None

        # 4. Process loop under inference_mode
        for img_path, f_name in zip(img_paths_list, file_names_list):
            # Load and preprocess image if present
            images_vl = []
            if img_path is not None:
                image_tensor = load_vlm_image_tensor(img_path)

                # Image resolution downscaling helper
                def process_vlm_image(img, res):
                    VLM_RESOLUTIONS = {
                        "Fast (384)": 384,
                        "Balanced (512)": 512,
                        "Detailed (768)": 768,
                        "Large (1024)": 1024,
                        "X-Large (1280)": 1280,
                        "XX-Large (1536)": 1536
                    }
                    samples = img.movedim(-1, 1)
                    if res == "Original":
                        return img
                    else:
                        vlm_size = VLM_RESOLUTIONS[res]
                        total_vlm = vlm_size * vlm_size
                        scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                        width_vlm = round(samples.shape[3] * scale_by_vlm)
                        height_vlm = round(samples.shape[2] * scale_by_vlm)

                        s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                        return s_vlm.movedim(1, -1)

                processed_img = process_vlm_image(image_tensor, vlm_resolution)
                images_vl.append(processed_img)

            # Tokenize prompt using skip_template=True so no template wrapping is saved
            modified_prompt = prompt
            if img_path is not None:
                if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            # Prepend <|im_start|> to trigger skip_template internally
            modified_prompt = "<|im_start|>" + modified_prompt
            tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

            key_name = next(iter(tokens.keys()))
            token_list = tokens[key_name]
            # Slice off the first token (the <|im_start|> trigger token) to match skip_template behavior
            for i in range(len(token_list)):
                token_list[i] = token_list[i][1:]
            tokens_only = [[t[0] for t in b] for b in token_list]
            device = comfy.model_management.get_torch_device()

            with torch.inference_mode():
                embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, device)

                if slice_visual_tokens:
                    vis_start, vis_end = find_visual_token_range(tokens, embeds)
                    if vis_start < vis_end:
                        prefix = embeds[:, :vis_start, :]
                        suffix = embeds[:, vis_end:, :]
                        embeds_sliced = torch.cat([prefix, suffix], dim=1)
                    else:
                        embeds_sliced = embeds
                else:
                    embeds_sliced = embeds

                tensor_2d = embeds_sliced.squeeze(0).clone().cpu()

            state_dict = {key_name: tensor_2d}

            # Save the safetensors file (ensure nested directories exist)
            target_path = resolve_embedding_output_path(embeddings_dir, f"{f_name}.safetensors")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            state_dict_safe = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(state_dict_safe, target_path)

            last_state_dict = state_dict
            last_tensor_2d = tensor_2d

            # Clean VRAM loop references
            del embeds, embeds_sliced, tokens, tokens_only
            if img_path is not None:
                del image_tensor, processed_img

        # 5. Final VRAM release and soft_empty_cache
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(last_state_dict, last_tensor_2d)


class UC_Qwen3VLInputEmbeds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_Qwen3VLInputEmbeds",
            display_name="Qwen3-VL Unified Input Embeddings",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Input text prompt. Important: skips any template wrapping."),
                io.String.Input("image_paths", multiline=True, default="", placeholder="C:/paths/to/image1.png\nC:/paths/to/image2.png", tooltip="Line-separated list of paths to image files. Must map 1-to-1 with file_names."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.String.Input("file_names", multiline=True, default="", placeholder="bulbasaur\nivysaur", tooltip="Line-separated list of file names to save as (without .safetensors). Can include nested subfolders. Must map 1-to-1 with image_paths."),
                io.Boolean.Input(
                    "slice_visual_tokens",
                    default=False,
                    tooltip="If True, removes the first validated visual-token span. If False, preserves the full interleaved sequence.",
                ),
            ],
            outputs=[
                io.AnyType.Output("state_dict", tooltip="Dictionary structure: {key_name: tensor_2d} of shape [num_tokens, hidden_size]"),
                io.AnyType.Output("tensor_2d", tooltip="Raw PyTorch 2D tensor of shape [num_tokens, hidden_size]"),
            ]
        )

    @classmethod
    def execute(cls, clip, prompt, image_paths, vlm_resolution, file_names, slice_visual_tokens=False) -> io.NodeOutput:
        # 1. Parse image paths and file names
        img_paths_list = [p.strip() for p in image_paths.split("\n") if p.strip()]
        file_names_list = [n.strip() for n in file_names.split("\n") if n.strip()]

        if not img_paths_list and not file_names_list:
            raise ValueError("Both image_paths and file_names are empty.")

        # If only text prompt encoding is desired (no images)
        if not img_paths_list:
            if not file_names_list:
                raise ValueError("No file_names specified to save the text embedding.")
            img_paths_list = [None] * len(file_names_list)
        elif not file_names_list:
            raise ValueError("No file_names specified for the provided image paths.")

        if len(img_paths_list) != len(file_names_list):
            raise ValueError(f"Count mismatch: Got {len(img_paths_list)} image paths and {len(file_names_list)} file names.")

        # 2. Pre-Execution Path Validation
        for path in img_paths_list:
            if path is not None:
                normalized_path = path.strip().replace('\\', '/')
                normalized_path = os.path.normpath(normalized_path)
                if not os.path.isabs(normalized_path):
                    normalized_path = os.path.abspath(normalized_path)
                if not os.path.isfile(normalized_path):
                    raise FileNotFoundError(
                        f"Validation aborted: Image file does not exist: '{path}' (resolved to: '{normalized_path}'). "
                        "No processing has started, ensuring safe memory state."
                    )

        # 3. Call clip.load_model() once to register the model as active for comfy-aimdo
        clip.load_model()
        cond_stage = clip.cond_stage_model
        clip_model = None

        if hasattr(cond_stage, "clip") and isinstance(cond_stage.clip, str) and hasattr(cond_stage, cond_stage.clip):
            clip_model = getattr(cond_stage, cond_stage.clip)
        elif hasattr(cond_stage, "clip_model"):
            clip_model = cond_stage.clip_model
        elif hasattr(cond_stage, "clip_d"):
            clip_model = cond_stage.clip_d
        else:
            clip_model = cond_stage

        if clip_model is None or not hasattr(clip_model, "process_tokens"):
            raise AttributeError("Could not locate underlying model wrapper with 'process_tokens' method in cond_stage_model.")


        # Locate ComfyUI embeddings directory
        try:
            embed_paths = folder_paths.get_folder_paths("embeddings")
            if embed_paths:
                embeddings_dir = embed_paths[0]
            else:
                embeddings_dir = os.path.join(os.path.dirname(folder_paths.__file__), "models", "embeddings")
        except Exception:
            embeddings_dir = "models/embeddings"

        os.makedirs(embeddings_dir, exist_ok=True)

        last_state_dict = None
        last_tensor_2d = None

        # 4. Process loop under inference_mode
        for img_path, f_name in zip(img_paths_list, file_names_list):
            # Load and preprocess image if present
            images_vl = []
            if img_path is not None:
                image_tensor = load_vlm_image_tensor(img_path)

                # Image resolution downscaling helper
                def process_vlm_image(img, res):
                    VLM_RESOLUTIONS = {
                        "Fast (384)": 384,
                        "Balanced (512)": 512,
                        "Detailed (768)": 768,
                        "Large (1024)": 1024,
                        "X-Large (1280)": 1280,
                        "XX-Large (1536)": 1536
                    }
                    samples = img.movedim(-1, 1)
                    if res == "Original":
                        return img
                    else:
                        vlm_size = VLM_RESOLUTIONS[res]
                        total_vlm = vlm_size * vlm_size
                        scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                        width_vlm = round(samples.shape[3] * scale_by_vlm)
                        height_vlm = round(samples.shape[2] * scale_by_vlm)

                        s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                        return s_vlm.movedim(1, -1)

                processed_img = process_vlm_image(image_tensor, vlm_resolution)
                images_vl.append(processed_img)

            # Tokenize prompt using skip_template=True so no template wrapping is saved
            modified_prompt = prompt
            if img_path is not None:
                if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            # Prepend <|im_start|> to trigger skip_template internally
            modified_prompt = "<|im_start|>" + modified_prompt
            tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

            # Retrieve the key name dynamically (typically "qwen3vl_4b" or "qwen3vl_8b")
            key_name = "qwen3vl_8b"
            if tokens:
                key_name = next(iter(tokens.keys()))
            token_list = tokens.get(key_name, [])
            # Slice off the first token (the <|im_start|> trigger token) to match skip_template behavior
            for i in range(len(token_list)):
                token_list[i] = token_list[i][1:]
            tokens_only = [[t[0] for t in b] for b in token_list]
            device = comfy.model_management.get_torch_device()

            with torch.inference_mode():
                embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, device)

                if slice_visual_tokens:
                    vis_start, vis_end = find_visual_token_range(tokens, embeds)
                    if vis_start < vis_end:
                        prefix = embeds[:, :vis_start, :]
                        suffix = embeds[:, vis_end:, :]
                        embeds_sliced = torch.cat([prefix, suffix], dim=1)
                    else:
                        embeds_sliced = embeds
                else:
                    embeds_sliced = embeds

                tensor_2d = embeds_sliced.squeeze(0).clone().cpu()

            state_dict = {key_name: tensor_2d}

            # Save the safetensors file (ensure nested directories exist)
            target_path = resolve_embedding_output_path(embeddings_dir, f"{f_name}.safetensors")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            state_dict_safe = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(state_dict_safe, target_path)

            last_state_dict = state_dict
            last_tensor_2d = tensor_2d

            # Clean VRAM loop references
            del embeds, embeds_sliced, tokens, tokens_only
            if img_path is not None:
                del image_tensor, processed_img

        # 5. Final VRAM release and soft_empty_cache
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(state_dict, tensor_2d)



_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END = 151644, 872, 198, 151645

class Krea2WeightPatch:
    def __get__(self, obj, objtype=None):
        return types.MethodType(krea2_attn_forward_weight, obj)

class TextEncodeKrea2SysEditScaledAdvAttn(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeKrea2SysEditScaledAdvAttn",
            display_name="Krea2 System Prompt Scaled Attention Encoder (Advanced)",
            category="advanced/conditioning",
            inputs=[
                # --- Primary Inputs ---
                io.Model.Input("model", tooltip="Diffusion model to apply the attention monkeypatch to."),
                io.Clip.Input("clip", tooltip="CLIP/T5 dual text encoder reference."),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main prompt. Fusion accepts image_input_fusion or image_input_1 for its single visual slot. Numbered multi-image inline placement is intentionally unavailable in this attention node.",
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default="", tooltip="System prompt injected prior to user description."),
                io.String.Input(
                    "attention_weights",
                    multiline=False,
                    default="",
                    tooltip="Space-separated non-negative attention odds weights. Example: (arms:1.5) (painting:0) (photo:2)",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Multimodal images. Maps active inputs sequentially to variables (a, b, c, ...)."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=4.0, step=0.05, tooltip="Global multiplier on the weighting effect. Effect compounds over all blocks."),

                # --- Modular Configurations ---
                VisualFusionConfig.Input("visual_fusion_config", optional=True, tooltip="Optional spatial visual fusion configuration from UC_VisualFusionConfig. Blends isolated visual blocks without coordinate blur."),

                # --- Fallback Controls ---
                io.String.Input(
                    "formula",
                    default="a",
                    multiline=False,
                    tooltip="Mathematical formula to blend conditioning outputs. Active ONLY if visual_fusion_config is disconnected or set to 'off'. Use variables a, b, c, d... to reference active connected image inputs.",
                ),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions. Active ONLY if visual_fusion_config is disconnected or set to 'off'.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, clip, prompt, system_prompt, attention_weights, image_inputs: io.Autogrow.Type, vlm_resolution: str, visual_fusion_config: dict = None, formula: str = "a", padding_method: str = "zero-pad", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, multiplier: float = 1.0, strength: float = 1.0) -> io.NodeOutput:
        # Collect, extract, and parse all active (non-null) connected images sequentially (including batched images)
        _, active_images, _ = extract_and_flatten_images(image_inputs)

        # 1. Parse weights from the attention_weights widget using regex
        pattern = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
        terms = [(m.group(1).strip(), float(m.group(2))) for m in pattern.finditer(attention_weights)]
        if any(not math.isfinite(weight) or weight < 0 for _, weight in terms):
            raise ValueError("Krea2 attention weights must be finite and non-negative.")

        weighted_prompt = prompt
        weighted_system_prompt = system_prompt

        def format_krea_prompt(user_text, system_text):
            if system_text:
                return (
                    "<|im_start|>user\n<|im_end|>\n"
                    f"<|im_start|>system\n{system_text}<|im_end|>\n"
                    f"<|im_start|>user\n{user_text}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            return (
                "<|im_start|>user\n<|im_end|>\n"
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        def add_image_marker(user_text):
            if any(tag in user_text for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                return user_text
            return "<|vision_start|><|image_pad|><|vision_end|>" + user_text

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "off"}
        visual_method = visual_fusion_config.get("visual_fusion_method", "off")

        if active_images and visual_method != "off":
            weighted_prompt, _ = prepare_image_placeholder_prompt(
                weighted_prompt,
                image_count=len(active_images),
                fusion_active=True,
                context="Krea2TokenAttentionWeight",
            )
        elif re.search(r"\bimage_input_(?:fusion|\d+)\b", weighted_prompt, re.IGNORECASE):
            logging.warning(
                "Krea2TokenAttentionWeight: numbered inline images are not supported because attention positions cannot be mapped safely across multiple visual spans; using the existing per-image path."
            )
            weighted_prompt, _ = prepare_image_placeholder_prompt(
                weighted_prompt,
                image_count=0,
                fusion_active=False,
                context="Krea2TokenAttentionWeight",
            )

        clean_prompt = strip_contextual_weight_syntax(weighted_prompt)
        clean_system_prompt = strip_contextual_weight_syntax(weighted_system_prompt)

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = common_upscale(samples, width_vlm, height_vlm, "bicubic", "disabled")
                return s_vlm.movedim(1, -1)

        # 2. Get tokens mapping on clean prompt with representative (first) image or fallback
        if active_images:
            first_img = active_images[0]
            processed_first_img = process_vlm_image(first_img, vlm_resolution)

            modified_clean_prompt = add_image_marker(clean_prompt)
            modified_weighted_prompt = add_image_marker(weighted_prompt)
            clean_full_prompt = format_krea_prompt(modified_clean_prompt, clean_system_prompt)
            weighted_full_prompt = format_krea_prompt(modified_weighted_prompt, weighted_system_prompt)
            tok = clip.tokenize(clean_full_prompt, images=[processed_first_img], skip_template=True)
        else:
            clean_full_prompt = format_krea_prompt(clean_prompt, clean_system_prompt)
            weighted_full_prompt = format_krea_prompt(weighted_prompt, weighted_system_prompt)
            tok = clip.tokenize(clean_full_prompt, skip_template=True)

        key = next(iter(tok))
        token_list = tok[key][0]
        ids = []
        for t in token_list:
            if isinstance(t, tuple) and len(t) > 0:
                ids.append(t[0])
            elif isinstance(t, dict):
                ids.append(-1)
            else:
                ids.append(t)

        cond = clip.encode_from_tokens_scheduled(tok)
        cond_len = cond[0][0].shape[1]

        mapping = build_token_to_conditioning_map(token_list, cond[0][0])

        weight_pairs = []
        for phrase, w in terms:
            k_bias = math.log(max(w, 1e-6)) * strength
            positions = []
            for variant in (" " + phrase, phrase):
                sub = krea2_token_ids(clip, variant)
                ps, pe = krea2_user_content_span(sub)
                if ps is not None:
                    sub = sub[ps:pe]
                matches = find_subsequence(ids, sub, 0, len(ids))
                if matches:
                    for mi in matches:
                        for off in range(len(sub)):
                            t_idx = mi + off
                            if t_idx < len(mapping):
                                positions.append(mapping[t_idx][0])
                    break
            if not positions:
                logging.warning(f"Krea2PromptWeight: phrase '{phrase}' not found in prompt or system prompt; skipped.")
                continue
            for cp in positions:
                if 0 <= cp < cond_len:
                    weight_pairs.append((cp, k_bias))

        # 3. Patch model
        model_clone = model.clone()
        if weight_pairs:
            logging.info(f"Krea2PromptWeight (Attn): weighting {weight_pairs}")
            diffusion_model = model_clone.get_model_object("diffusion_model")
            transformer_options = model_clone.model_options.get("transformer_options", {}).copy()
            transformer_options["krea2_token_weights"] = weight_pairs
            model_clone.model_options["transformer_options"] = transformer_options

            for idx, block in enumerate(diffusion_model.blocks):
                if hasattr(block, "attn"):
                    patched_attn = Krea2WeightPatch().__get__(block.attn, block.attn.__class__)
                    model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.attn.forward", patched_attn)

        # 4. Multipass encoding and blending
        if active_images:
            sequence_tensors = {}
            pooled_tensors = {}
            visual_ranges = {}
            tokens_dict = {}
            reference_cond_dict = None

            for idx, img in enumerate(active_images):
                letter = chr(97 + idx)
                processed_img = process_vlm_image(img, vlm_resolution)

                clean_pass_prompt = format_krea_prompt(add_image_marker(clean_prompt), clean_system_prompt)
                weighted_pass_prompt = format_krea_prompt(add_image_marker(weighted_prompt), weighted_system_prompt)

                cond_X = encode_embedding_classical_scaled_bias(clip, weighted_pass_prompt, images=[processed_img], skip_template=True)
                if len(cond_X) != 1:
                    raise ValueError("Krea2 attention visual fusion requires a single conditioning schedule entry.")
                C_X = cond_X[0][0]
                P_X = cond_X[0][1].get("pooled_output", None)

                sequence_tensors[letter] = C_X
                if P_X is not None:
                    pooled_tensors[letter] = P_X

                if visual_method != "off":
                    try:
                        tokens = clip.tokenize(clean_pass_prompt, images=[processed_img], skip_template=True)
                        tokens_dict[letter] = tokens
                        vis_start, vis_end = find_visual_token_range(tokens, C_X)
                        visual_ranges[letter] = (vis_start, vis_end)
                    except Exception as e:
                        raise ValueError(f"Could not locate the visual token range for image {idx + 1}: {e}") from e

                if reference_cond_dict is None:
                    reference_cond_dict = cond_X[0][1]

            fusion_mask_cache = {}
            if visual_method != "off":
                device = comfy.model_management.get_torch_device()

                key_name = "qwen3vl_8b"
                if "tokens" in locals() and tokens:
                    key_name = next(iter(tokens.keys()))

                C_blended, P_blended = evaluate_conditioning_consensus_blend(
                    sequence_tensors, pooled_tensors, visual_fusion_config=visual_fusion_config, device=device, visual_ranges=visual_ranges, embedding_key=key_name, clip=clip, tokens_dict=tokens_dict, mask_cache=fusion_mask_cache
                )
            else:
                C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

            # Build final conditioning dictionary
            final_cond_dict = reference_cond_dict.copy()
            attention_mask = final_cond_dict.get("attention_mask")
            if torch.is_tensor(attention_mask) and attention_mask.shape[-1] != C_blended.shape[1]:
                final_cond_dict.pop("attention_mask", None)
                final_cond_dict.pop("attention_mask_img_shape", None)
            if P_blended is not None:
                final_cond_dict["pooled_output"] = P_blended

            if multiplier != 1.0:
                C_blended *= multiplier
                if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                    final_cond_dict["pooled_output"] *= multiplier

            conditioning = [[C_blended, final_cond_dict]]
        else:
            conditioning = encode_embedding_classical_scaled_bias(clip, weighted_full_prompt, skip_template=True)
            if multiplier != 1.0:
                for i in range(len(conditioning)):
                    conditioning[i][0] *= multiplier
                    if "pooled_output" in conditioning[i][1] and conditioning[i][1]["pooled_output"] is not None:
                        conditioning[i][1]["pooled_output"] *= multiplier

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process sequentially from active_images
            for i, image in enumerate(active_images):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                if image is not None:
                    samples = image.movedim(-1, 1)
                    if vae_resolution == "Original":
                        width_vae = round(samples.shape[3] / 8.0) * 8
                        height_vae = round(samples.shape[2] / 8.0) * 8
                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
                    else:
                        vae_size = VAE_RESOLUTIONS[vae_resolution]
                        total_vae = vae_size * vae_size
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8

                        s_vae = common_upscale(samples, width_vae, height_vae, "bicubic", "disabled")
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)

        return io.NodeOutput(model_clone, conditioning)


# Canonical 0.10 node IDs. Compatibility classes below remain registered for
# one release and delegate to the same execution implementations.
class UC_TextEncodeSystemEditAdvanced(TextEncodeSystemEditPlusAdvanced):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_TextEncodeSystemEditAdvanced"
        schema.display_name = "System Edit Text Encode (Advanced)"
        schema.is_deprecated = False
        return schema


class UC_TextEncodeGemmaSystemEditAdvanced(TextEncodeGemmaSystemEditPlusAdvanced):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_TextEncodeGemmaSystemEditAdvanced"
        schema.display_name = "Gemma System Edit Text Encode (Advanced)"
        schema.is_deprecated = False
        return schema


class UC_AdvancedVisualConditioningEncode(TextEncodeKrea2SystemEditScaledAdv):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_AdvancedVisualConditioningEncode"
        schema.display_name = "Advanced Visual Conditioning Encode"
        schema.is_deprecated = False
        # Model-backed validation is required before this can be stable.
        schema.is_experimental = True
        return schema


class UC_VLMInputEmbeds(UC_Qwen3VLInputEmbeds):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_VLMInputEmbeds"
        schema.display_name = "VLM Input Embedding Export"
        schema.is_deprecated = False
        return schema


class UC_Krea2TokenAttentionWeight(TextEncodeKrea2SysEditScaledAdvAttn):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_Krea2TokenAttentionWeight"
        schema.display_name = "Krea2 Token Attention Weight"
        schema.is_deprecated = False
        schema.is_experimental = True
        return schema


def _mark_deprecated_node(node_class):
    original = node_class.define_schema.__func__

    @classmethod
    def deprecated_schema(cls):
        schema = original(cls)
        schema.is_deprecated = True
        return schema

    node_class.define_schema = deprecated_schema


for _deprecated_node in (
    UC_TextEncodeFlux2SystemPrompt,
    UC_TextEncodeKleinSystemPrompt,
    UC_TextEncodeKrea2SystemPrompt,
    UC_TextEncodeZITSystemPrompt,
    UC_TextEncodeZImageThinkPrompt,
    UC_ScaledBiasTextEncodeFlux2SystemPrompt,
    UC_ScaledBiasTextEncodeKleinSystemPrompt,
    UC_ScaledBiasTextEncodeLtxv2SystemPrompt,
    UC_ScaledBiasTextEncodeZITSystemPrompt,
    UC_ScaledBiasTextEncodeZImageThinkPrompt,
    UC_ScaledBiasTextEncodeSystemPrompt,
    TextEncodeSystemEditPlus,
    TextEncodeSystemEditPlusAdvanced,
    TextEncodeKrea2SystemEditPlusAdvanced,
    TextEncodeEditPlusAdvanced,
    TextEncodeKrea2SystemEditScaledAdv,
    TextEncodeEditScaledAdv,
    TextEncodeGemmaSystemEditPlusAdvanced,
    UC_Krea2InputEmbeds,
    UC_Qwen3VLInputEmbeds,
    TextEncodeKrea2SysEditScaledAdvAttn,
):
    _mark_deprecated_node(_deprecated_node)


