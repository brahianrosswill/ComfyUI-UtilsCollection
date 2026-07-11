import re
import torch
import math
import os

from comfy_api.latest import ComfyExtension, io
from comfy.utils import common_upscale
import node_helpers
from .helper_functions import get_token_count, get_token_count_scaled
from .encoder_helpers import(
    encode_embedding_scaled_bias,
    is_image_token,
    evaluate_formula,
    evaluate_conditioning_formula,
    reconstruct_2d_grid,
    generate_spatial_fusion_mask,
    save_blended_visual_embeddings,
    evaluate_conditioning_consensus_blend,
    blend_text_vectors,
    find_visual_token_range,
    encode_embedding_classical_scaled_bias,
    load_vlm_image_tensor,
    krea2_user_content_span,
    krea2_token_ids,
    find_subsequence,
    krea2_attn_forward_weight,
)

def apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode):
    if not ref_latents:
        return conditioning

    if "parallel" in ref_latent_mode:
        import comfy
        # Encode empty prompt as neutral base
        tokens_neutral = clip.tokenize("")
        conditioning_neutral = clip.encode_from_tokens_scheduled(tokens_neutral)

        out = []
        for i in range(len(conditioning)):
            c_vlm, meta_vlm = conditioning[i]
            c_neutral, meta_neutral = conditioning_neutral[i] if i < len(conditioning_neutral) else conditioning_neutral[-1]

            c_neutral_cast = comfy.model_management.cast_to_device(c_neutral, c_vlm.device, c_vlm.dtype)
            c_combined = torch.cat([c_vlm, c_neutral_cast], dim=1)

            meta_combined = meta_vlm.copy()
            meta_combined["reference_latents"] = ref_latents
            if "attention_mask" in meta_combined:
                del meta_combined["attention_mask"]

            out.append([c_combined, meta_combined])
        return out
    else:
        # Standard append mode
        return node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)


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
            ]
        )

    @classmethod
    def execute(cls, clip, text) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        if '<' not in text and '>' not in text and '=' not in text:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return ([[cond, {"pooled_output": pooled}]], )

        bias_pattern = re.compile(r"<([^>]+)=([0-9.-]+)>")
        split_pattern = re.compile(r"(<[^>]+=[0-9.-]+>)")
        segments = split_pattern.split(text)

        clean_text = ""
        biases_to_apply = []

        current_token_index = 1

        for segment in segments:
            if not segment:
                continue

            match = bias_pattern.fullmatch(segment)
            if match:
                bias_text, strength_str = match.groups()
                strength = float(strength_str)
                clean_text += bias_text
                num_tokens = get_token_count(clip, bias_text)

                if num_tokens > 0:
                    start_index = current_token_index
                    end_index = current_token_index + num_tokens
                    biases_to_apply.append({"start": start_index, "end": end_index, "strength": strength})

                current_token_index += num_tokens
            else:
                clean_text += segment
                num_tokens = get_token_count(clip, segment)
                current_token_index += num_tokens

        tokens = clip.tokenize(clean_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        if not biases_to_apply:
            return ([[cond, {"pooled_output": pooled}]], )

        cond_dict = {"pooled_output": pooled}
        n_text_tokens = cond.shape[1]
        device = cond.device
        dtype = torch.float16


        final_seq_len = n_text_tokens + 1
        attn_mask = torch.zeros((1, final_seq_len, final_seq_len), dtype=dtype, device=device)

        pooled_offset = 1

        for bias in biases_to_apply:
            strength = bias["strength"]
            attn_bias_value = torch.log(torch.tensor(strength, dtype=dtype, device=device))

            start = min(bias["start"] + pooled_offset, final_seq_len)
            end = min(bias["end"] + pooled_offset, final_seq_len)

            if start >= end:
                continue

            attn_mask[:, :, start:end] += attn_bias_value
            attn_mask[:, start:end, :] += attn_bias_value

        cond_dict["attention_mask"] = attn_mask
        cond_dict["attention_mask_img_shape"] = (1, 1)


        new_conditioning = ([[cond, cond_dict]])

        return io.NodeOutput(new_conditioning)

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
            ]
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
                    options=["off", "linear", "index-consensus", "similarity-consensus", "spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"],
                    default="spatial-checkerboard",
                    tooltip="Method to combine isolated visual tokens. Methods starting with 'spatial-' interleave pure token vectors to keep details perfectly sharp and achieve true fusion instead of filter-like blurring."
                ),
                io.Int.Input("visual_block_size", default=2, min=1, max=8, step=1, tooltip="Active for spatial-block-interleave. Size of the spatial token patches to group and switch together."),
                io.Float.Input("dither_ratio", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Active for spatial-dither-random. Probability of selecting a token from the first image vs subsequent images."),
                io.Boolean.Input("save_blended_embeds", default=False, tooltip="Enable to save the blended visual tokens as a standalone .safetensors embedding."),
                io.String.Input("save_path", default="blended_visual_embeds.safetensors", tooltip="Target filename/path under models/embeddings to save the .safetensors file.")
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
        save_path: str = "blended_visual_embeds.safetensors"
    ) -> io.NodeOutput:
        config = {
            "visual_fusion_method": visual_fusion_method,
            "visual_block_size": visual_block_size,
            "dither_ratio": dither_ratio,
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
            ]
        )

    @classmethod
    def execute(cls, conditioning_inputs: io.Autogrow.Type, text_blend_config: dict = None) -> io.NodeOutput:
        """
        Blends stock or custom ComfyUI conditioning outputs post-encoder using CWB math.
        """
        import torch
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

        # Extract sequence and pooled tensors across conditionings
        sequence_tensors = {}
        pooled_tensors = {}

        # ComfyUI conditioning data structure is a list of lists of dictionaries:
        # [ [ [conditioning_tensor, { "pooled_output": pooled_tensor, ... }] ] ]
        for idx, cond_list in enumerate(active_conds):
            key = chr(97 + idx) # 'a', 'b', 'c', ...
            # Extract first element's tensor
            main_tensor = cond_list[0][0]
            sequence_tensors[key] = main_tensor

            meta_dict = cond_list[0][1]
            if meta_dict and "pooled_output" in meta_dict:
                pooled_tensors[key] = meta_dict["pooled_output"]
            else:
                pooled_tensors[key] = None

        # Execute CWB blending
        import comfy
        device = comfy.model_management.get_torch_device()
        dtype = sequence_tensors['a'].dtype

        # Cast all sequence and pooled tensors safely using Comfy's non-blocking, aimdo-aware pipeline
        safe_sequence_tensors = {}
        for k, t in sequence_tensors.items():
            safe_sequence_tensors[k] = comfy.model_management.cast_to_device(t, device, dtype)

        safe_pooled_tensors = {}
        for k, p in pooled_tensors.items():
            if p is not None:
                safe_pooled_tensors[k] = comfy.model_management.cast_to_device(p, device, dtype)
            else:
                safe_pooled_tensors[k] = None

        C_blended, P_blended = blend_text_vectors(
            safe_sequence_tensors,
            text_blend_config,
            pooled_tensors=safe_pooled_tensors,
            device=str(device)
        )

        # Reconstruct ComfyUI conditioning list structure
        # Clone reference meta dict to preserve auxiliary items (like guidance, controls, etc.)
        ref_meta = active_conds[0][0][1].copy()
        if P_blended is not None:
            ref_meta["pooled_output"] = P_blended
        elif "pooled_output" in ref_meta:
            del ref_meta["pooled_output"]

        blended_conditioning = [[C_blended, ref_meta]]
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
        # Collect and parse all autogrow keys
        raw_images = {}
        if image_inputs is not None:
            # image_inputs can be a dict, let's handle cases where it might be empty
            for k, v in image_inputs.items():
                if v is not None:
                    # Extract numeric suffix (e.g. "image1" -> 1)
                    digits = re.findall(r'\d+', k)
                    if digits:
                        idx = int(digits[0])
                    else:
                        idx = 1
                    raw_images[idx] = v

        # Determine indexing: 0-indexed or 1-indexed.
        is_zero_indexed = 0 in raw_images

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
        # Collect and parse all autogrow keys
        raw_images = {}
        if image_inputs is not None:
            # image_inputs can be a dict, let's handle cases where it might be empty
            for k, v in image_inputs.items():
                if v is not None:
                    # Extract numeric suffix (e.g. "image1" -> 1)
                    digits = re.findall(r'\d+', k)
                    if digits:
                        idx = int(digits[0])
                    else:
                        idx = 1
                    raw_images[idx] = v

        # Determine indexing: 0-indexed or 1-indexed.
        is_zero_indexed = 0 in raw_images

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
        # Collect and parse all autogrow keys
        raw_images = {}
        if image_inputs is not None:
            # image_inputs can be a dict, let's handle cases where it might be empty
            for k, v in image_inputs.items():
                if v is not None:
                    # Extract numeric suffix (e.g. "image1" -> 1)
                    digits = re.findall(r'\d+', k)
                    if digits:
                        idx = int(digits[0])
                    else:
                        idx = 1
                    raw_images[idx] = v

        # Determine indexing: 0-indexed or 1-indexed.
        is_zero_indexed = 0 in raw_images

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
        # Collect and parse all autogrow keys
        raw_images = {}
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    # Extract numeric suffix (e.g. "image1" -> 1)
                    digits = re.findall(r'\d+', k)
                    if digits:
                        idx = int(digits[0])
                    else:
                        idx = 1
                    raw_images[idx] = v

        # Determine indexing: 0-indexed or 1-indexed.
        is_zero_indexed = 0 in raw_images

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
}


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
            tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=True)
        elif len(system_prompt) > 0:
            template = SYSTEM_PROMPT_TEMPLATES.get(model_type, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
            llama_template = f"{template['prefix']}{system_prompt}{template['suffix']}"
            # If klein was chosen but without custom thinking_content, it uses the SYSTEM_PROMPT_TEMPLATES definition
            # which has an empty thinking block pre-defined inside suffix: "<think>\n\n</think>\n\n".
            # We must skip template to prevent the core from appending another redundant think block!
            skip_template = (model_type == "klein")
            tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=skip_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)




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
                    tooltip="Main user text prompt. Supports classical weight syntax: (prompt:weight), e.g. (sunset:1.2).",
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
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, visual_fusion_config: dict = None, formula: str = "a", padding_method: str = "zero-pad", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, multiplier: float = 1.0) -> io.NodeOutput:
        # Collect and parse all active (non-null) connected images sequentially
        active_images = []
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    active_images.append(v)

        if not active_images:
            # Fallback if no images are connected: encode prompt as plain text
            conditioning = encode_embedding_classical_scaled_bias(clip, prompt, multiplier=multiplier)
            return io.NodeOutput(conditioning)

        # Map active images sequentially to letter variables (a, b, c, ...) and encode each pass
        sequence_tensors = {}
        pooled_tensors = {}
        deepstack_dict = {}
        visual_ranges = {}
        tokens_dict = {}
        ref_cond_dict = None
        last_cond_dict = None

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "spatial-checkerboard", "visual_block_size": 2, "dither_ratio": 0.5}
        visual_method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")

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

            # Wrap in Llama/Krea2 system prompt template format
            if len(system_prompt) > 0:
                full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                    "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            else:
                full_prompt = (
                    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                    "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )

            # Encode individual sequence pass
            cond_X = encode_embedding_classical_scaled_bias(clip, full_prompt, images=[processed_img], skip_template=True)
            C_X = cond_X[0][0]
            P_X = cond_X[0][1].get("pooled_output", None)

            sequence_tensors[letter] = C_X
            if P_X is not None:
                pooled_tensors[letter] = P_X

            # Tokenize and find visual token range for isolation blending
            try:
                tokens = clip.tokenize(full_prompt, images=[processed_img], skip_template=True)
                tokens_dict[letter] = tokens
                vis_start, vis_end = find_visual_token_range(tokens, C_X)
                visual_ranges[letter] = (vis_start, vis_end)
            except Exception:
                visual_ranges[letter] = (0, 0)

            # Extract DeepStack per-layer tensors if present
            if "embeds_info" in cond_X[0][1] and len(cond_X[0][1]["embeds_info"]) > 0:
                extra = cond_X[0][1]["embeds_info"][0].get("extra", {})
                if "deepstack" in extra:
                    deepstack_dict[letter] = extra["deepstack"]

            if idx == 0:
                ref_cond_dict = cond_X[0][1]
            last_cond_dict = cond_X[0][1]

        # Evaluate mathematical formula or consensus on sequence and pooled tensors
        if visual_method != "off":
            import comfy
            device = comfy.model_management.get_torch_device()

            key_name = "qwen3vl_8b"
            if "tokens" in locals() and tokens:
                key_name = next(iter(tokens.keys()))

            C_blended, P_blended = evaluate_conditioning_consensus_blend(
                sequence_tensors, pooled_tensors, visual_fusion_config=visual_fusion_config, device=device, visual_ranges=visual_ranges, embedding_key=key_name, clip=clip, tokens_dict=tokens_dict
            )
        else:
            C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

        # Evaluate mathematical formula on DeepStack layers
        deepstack_blended = None
        if deepstack_dict:
            first_key = next(iter(sequence_tensors.keys()))
            num_layers = len(deepstack_dict[first_key])
            max_vis_len = max(ds_list[0].shape[0] for ds_list in deepstack_dict.values())

            deepstack_blended = []
            import torch.nn.functional as F

            for l in range(num_layers):
                layer_tensors = {let: ds_list[l] for let, ds_list in deepstack_dict.items()}

                if visual_method != "off":
                    wrapped_layer_tensors = {let: t.unsqueeze(0) for let, t in layer_tensors.items()}
                    import comfy
                    device = comfy.model_management.get_torch_device()
                    # DeepStack intermediate layers must be blended recursively. We use linear blending fallback.
                    C_l_blended, _ = evaluate_conditioning_consensus_blend(
                        wrapped_layer_tensors, {}, visual_fusion_config={"visual_fusion_method": "linear"}, device=device
                    )
                    deepstack_blended.append(C_l_blended.squeeze(0))
                else:
                    # Align layer tensors to maximum length
                    aligned_layer_tensors = {}
                    for name, tensor in layer_tensors.items():
                        if tensor.shape[0] < max_vis_len:
                            if padding_method == "interpolate":
                                tensor_perm = tensor.permute(1, 0).unsqueeze(0)
                                tensor_interp = F.interpolate(tensor_perm, size=max_vis_len, mode='linear', align_corners=False)
                                tensor = tensor_interp.squeeze(0).permute(1, 0)
                            else:  # zero-pad
                                pad_size = max_vis_len - tensor.shape[0]
                                padding = torch.zeros((pad_size, tensor.shape[1]), device=tensor.device, dtype=tensor.dtype)
                                tensor = torch.cat([tensor, padding], dim=0)
                        aligned_layer_tensors[name] = tensor

                    safe_dict_layer = {
                        "__builtins__": {},
                        "clamp": torch.clamp,
                        "min": torch.minimum,
                        "max": torch.maximum,
                        "abs": torch.abs,
                    }
                    for name, t in aligned_layer_tensors.items():
                        safe_dict_layer[name] = t

                    # Handle classical weighting conversions inside math formula
                    layer_expression = re.sub(
                        r"\(\s*([a-zA-Z0-9_]+)\s*:\s*([0-9.-]+)\s*\)",
                        r"(\1 * \2)",
                        formula
                    )

                    try:
                        layer_blended = eval(layer_expression, safe_dict_layer, {}) # noqa: S307
                        deepstack_blended.append(layer_blended)
                    except Exception as e:
                        raise RuntimeError(f"Error evaluating DeepStack math expression at layer {l}: {e}")

        # Build final conditioning dictionary
        final_cond_dict = last_cond_dict.copy()
        if P_blended is not None:
            final_cond_dict["pooled_output"] = P_blended

        if deepstack_blended is not None and "embeds_info" in final_cond_dict and len(final_cond_dict["embeds_info"]) > 0:
            final_cond_dict["embeds_info"] = [final_cond_dict["embeds_info"][0].copy()]
            final_cond_dict["embeds_info"][0]["extra"] = final_cond_dict["embeds_info"][0]["extra"].copy()
            final_cond_dict["embeds_info"][0]["extra"]["deepstack"] = deepstack_blended

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
        # Collect and parse all active (non-null) connected images sequentially
        active_images = []
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    active_images.append(v)

        if not active_images:
            # Fallback if no images are connected: encode prompt as plain text
            conditioning = encode_embedding_classical_scaled_bias(clip, prompt, multiplier=multiplier)
            return io.NodeOutput(conditioning)

        # Map active images sequentially to letter variables (a, b, c, ...) and encode each pass
        sequence_tensors = {}
        pooled_tensors = {}
        visual_ranges = {}
        tokens_dict = {}
        ref_cond_dict = None
        last_cond_dict = None

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "spatial-checkerboard", "visual_block_size": 2, "dither_ratio": 0.5}
        visual_method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")

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
            C_X = cond_X[0][0]
            P_X = cond_X[0][1].get("pooled_output", None)

            sequence_tensors[letter] = C_X
            if P_X is not None:
                pooled_tensors[letter] = P_X

            # Tokenize and find visual token range for isolation blending
            try:
                tokens = clip.tokenize(modified_prompt, images=[processed_img], skip_template=True)
                tokens_dict[letter] = tokens
                vis_start, vis_end = find_visual_token_range(tokens, C_X)
                visual_ranges[letter] = (vis_start, vis_end)
            except Exception:
                visual_ranges[letter] = (0, 0)

            if idx == 0:
                ref_cond_dict = cond_X[0][1]
            last_cond_dict = cond_X[0][1]

        # Evaluate mathematical formula or consensus on sequence and pooled tensors
        if visual_method != "off":
            import comfy
            device = comfy.model_management.get_torch_device()

            key_name = "qwen3vl_8b"
            if "tokens" in locals() and tokens:
                key_name = next(iter(tokens.keys()))

            C_blended, P_blended = evaluate_conditioning_consensus_blend(
                sequence_tensors, pooled_tensors, visual_fusion_config=visual_fusion_config, device=device, visual_ranges=visual_ranges, embedding_key=key_name, clip=clip, tokens_dict=tokens_dict
            )
        else:
            C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

        # Build final conditioning dictionary
        final_cond_dict = last_cond_dict.copy()
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
                    tooltip="If True, performs perfect visual slicing (Method A) to cut out visual tokens, saving a pure language embedding. If False (default), preserves the full interleaved sequence including visual tokens.",
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

        import folder_paths
        from safetensors.torch import save_file

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

            tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

            key_name = next(iter(tokens.keys()))
            token_list = tokens[key_name]
            tokens_only = [[t[0] for t in b] for b in token_list]

            import comfy
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
            target_path = os.path.join(embeddings_dir, f"{f_name}.safetensors")
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
        import gc
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
                    tooltip="If True, performs perfect visual slicing (Method A) to cut out visual tokens, saving a pure language embedding. If False (default), preserves the full interleaved sequence including visual tokens.",
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

        import folder_paths
        from safetensors.torch import save_file

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

            tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

            # Retrieve the key name dynamically (typically "qwen3vl_4b" or "qwen3vl_8b")
            key_name = "qwen3vl_8b"
            if tokens:
                key_name = next(iter(tokens.keys()))
            token_list = tokens.get(key_name, [])
            tokens_only = [[t[0] for t in b] for b in token_list]

            import comfy
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
            target_path = os.path.join(embeddings_dir, f"{f_name}.safetensors")
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
        import gc
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(state_dict, tensor_2d)


_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END = 151644, 872, 198, 151645

class Krea2WeightPatch:
    def __get__(self, obj, objtype=None):
        import types
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
                    tooltip="Main user text prompt.",
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default="", tooltip="System prompt injected prior to user description."),
                io.String.Input(
                    "attention_weights",
                    multiline=False,
                    default="",
                    tooltip="Space-separated list of weighted words/phrases. Example: (arms:1.5) (painting:-1) (photo:2)",
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
        )

    @classmethod
    def execute(cls, model, clip, prompt, system_prompt, attention_weights, image_inputs: io.Autogrow.Type, vlm_resolution: str, visual_fusion_config: dict = None, formula: str = "a", padding_method: str = "zero-pad", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, multiplier: float = 1.0, strength: float = 1.0) -> io.NodeOutput:
        active_images = []
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    active_images.append(v)

        # 1. Parse weights from the attention_weights widget using regex
        import re
        pattern = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
        terms = [(m.group(1).strip(), float(m.group(2))) for m in pattern.finditer(attention_weights)]

        # Prompt inputs remain as untouched plain-text strings
        clean_prompt = prompt
        clean_system_prompt = system_prompt

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "spatial-checkerboard", "visual_block_size": 2, "dither_ratio": 0.5}
        visual_method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")

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

            modified_clean_prompt = clean_prompt
            if not any(tag in clean_prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                modified_clean_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_clean_prompt

            if len(clean_system_prompt) > 0:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\n" + clean_system_prompt + "<|im_end|>\n" +
                    "<|im_start|>user\n" + modified_clean_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            else:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                    "<|im_start|>user\n" + modified_clean_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            tok = clip.tokenize(clean_full_prompt, images=[processed_first_img], skip_template=True)
        else:
            if len(clean_system_prompt) > 0:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\n" + clean_system_prompt + "<|im_end|>\n" +
                    "<|im_start|>user\n" + clean_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
            else:
                clean_full_prompt = (
                    "<|im_start|>user\n" + "<|im_end|>\n" +
                    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                    "<|im_start|>user\n" + clean_prompt + "<|im_end|>\n" +
                    "<|im_start|>assistant\n"
                )
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

        # Count text vs image tokens for mapping
        text_token_count = 0
        image_token_count = 0
        for t in token_list:
            if is_image_token(t):
                image_token_count += 1
            else:
                text_token_count += 1

        if image_token_count > 0:
            V = (cond_len - text_token_count) // image_token_count
        else:
            V = 1

        mapping = []
        current_idx = 0
        for t in token_list:
            is_img = is_image_token(t)
            size = V if is_img else 1
            start = current_idx
            end = current_idx + size
            mapping.append((start, end))
            current_idx = end

        weight_pairs = []
        for phrase, w in terms:
            if w > 1.0:
                v_factor, k_bias = 1.0, strength * (w - 1.0) * 2.0
            else:
                v_factor, k_bias = 1.0 + strength * (w - 1.0), 0.0
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
                import logging
                logging.warning(f"Krea2PromptWeight: phrase '{phrase}' not found in prompt or system prompt; skipped.")
                continue
            for cp in positions:
                if 0 <= cp < cond_len:
                    weight_pairs.append((cp, v_factor, k_bias))

        # 3. Patch model
        model_clone = model.clone()
        if weight_pairs:
            import logging
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
            deepstack_dict = {}
            visual_ranges = {}
            tokens_dict = {}
            ref_cond_dict = None
            last_cond_dict = None

            for idx, img in enumerate(active_images):
                letter = chr(97 + idx)
                processed_img = process_vlm_image(img, vlm_resolution)

                modified_prompt = clean_prompt
                if not any(tag in clean_prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

                if len(clean_system_prompt) > 0:
                    full_prompt = (
                        "<|im_start|>user\n" + "<|im_end|>\n" +
                        "<|im_start|>system\n" + clean_system_prompt + "<|im_end|>\n" +
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

                cond_X = encode_embedding_classical_scaled_bias(clip, full_prompt, images=[processed_img], skip_template=True)
                C_X = cond_X[0][0]
                P_X = cond_X[0][1].get("pooled_output", None)

                sequence_tensors[letter] = C_X
                if P_X is not None:
                    pooled_tensors[letter] = P_X

                # Tokenize and find visual token range for isolation blending
                try:
                    tokens = clip.tokenize(full_prompt, images=[processed_img], skip_template=True)
                    tokens_dict[letter] = tokens
                    vis_start, vis_end = find_visual_token_range(tokens, C_X)
                    visual_ranges[letter] = (vis_start, vis_end)
                except Exception:
                    visual_ranges[letter] = (0, 0)

                if "embeds_info" in cond_X[0][1] and len(cond_X[0][1]["embeds_info"]) > 0:
                    extra = cond_X[0][1]["embeds_info"][0].get("extra", {})
                    if "deepstack" in extra:
                        deepstack_dict[letter] = extra["deepstack"]

                if idx == 0:
                    ref_cond_dict = cond_X[0][1]
                last_cond_dict = cond_X[0][1]

            if visual_method != "off":
                import comfy
                device = comfy.model_management.get_torch_device()

                key_name = "qwen3vl_8b"
                if "tokens" in locals() and tokens:
                    key_name = next(iter(tokens.keys()))

                C_blended, P_blended = evaluate_conditioning_consensus_blend(
                    sequence_tensors, pooled_tensors, visual_fusion_config=visual_fusion_config, device=device, visual_ranges=visual_ranges, embedding_key=key_name, clip=clip, tokens_dict=tokens_dict
                )
            else:
                C_blended, P_blended = evaluate_conditioning_formula(formula, sequence_tensors, pooled_tensors, padding_method=padding_method)

            deepstack_blended = None
            if deepstack_dict:
                first_key = next(iter(sequence_tensors.keys()))
                num_layers = len(deepstack_dict[first_key])
                max_vis_len = max(ds_list[0].shape[0] for ds_list in deepstack_dict.values())

                deepstack_blended = []
                import torch.nn.functional as F

                for l in range(num_layers):
                    layer_tensors = {let: ds_list[l] for let, ds_list in deepstack_dict.items()}

                    if visual_method != "off":
                        wrapped_layer_tensors = {let: t.unsqueeze(0) for let, t in layer_tensors.items()}
                        import comfy
                        device = comfy.model_management.get_torch_device()
                        # DeepStack intermediate layers are blended linearly for spatial stability
                        C_l_blended, _ = evaluate_conditioning_consensus_blend(
                            wrapped_layer_tensors, {}, visual_fusion_config={"visual_fusion_method": "linear"}, device=device
                        )
                        deepstack_blended.append(C_l_blended.squeeze(0))
                    else:
                        aligned_layer_tensors = {}
                        for name, tensor in layer_tensors.items():
                            if tensor.shape[0] < max_vis_len:
                                if padding_method == "interpolate":
                                    tensor_perm = tensor.permute(1, 0).unsqueeze(0)
                                    tensor_interp = F.interpolate(tensor_perm, size=max_vis_len, mode='linear', align_corners=False)
                                    tensor = tensor_interp.squeeze(0).permute(1, 0)
                                else:
                                    pad_size = max_vis_len - tensor.shape[0]
                                    padding = torch.zeros((pad_size, tensor.shape[1]), device=tensor.device, dtype=tensor.dtype)
                                    tensor = torch.cat([tensor, padding], dim=0)
                            aligned_layer_tensors[name] = tensor

                        safe_dict_layer = {
                            "__builtins__": {},
                            "clamp": torch.clamp,
                            "min": torch.minimum,
                            "max": torch.maximum,
                            "abs": torch.abs,
                        }
                        for name, t in aligned_layer_tensors.items():
                            safe_dict_layer[name] = t

                        layer_expression = re.sub(
                            r"\(\s*([a-zA-Z0-9_]+)\s*:\s*([0-9.-]+)\s*\)",
                            r"(\1 * \2)",
                            formula
                        )

                        try:
                            layer_blended = eval(layer_expression, safe_dict_layer, {})
                            deepstack_blended.append(layer_blended)
                        except Exception as e:
                            raise RuntimeError(f"Error evaluating DeepStack math expression at layer {l}: {e}")

            # Build final conditioning dictionary
            final_cond_dict = last_cond_dict.copy()
            if P_blended is not None:
                final_cond_dict["pooled_output"] = P_blended

            if deepstack_blended is not None and "embeds_info" in final_cond_dict and len(final_cond_dict["embeds_info"]) > 0:
                final_cond_dict["embeds_info"] = [final_cond_dict["embeds_info"][0].copy()]
                final_cond_dict["embeds_info"][0]["extra"] = final_cond_dict["embeds_info"][0]["extra"].copy()
                final_cond_dict["embeds_info"][0]["extra"]["deepstack"] = deepstack_blended

            if multiplier != 1.0:
                C_blended *= multiplier
                if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                    final_cond_dict["pooled_output"] *= multiplier

            conditioning = [[C_blended, final_cond_dict]]
        else:
            conditioning = encode_embedding_classical_scaled_bias(clip, clean_full_prompt)
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


