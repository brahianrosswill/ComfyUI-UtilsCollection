import pathlib
import sys
import types

import pytest
import numpy as np
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_encoder_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_encoder_test import encoder_helpers
    from utils_collection_encoder_test.encoder_nodes import (
        TextEncodeKrea2SysEditScaledAdvAttn,
        UC_AttentionBiasTextEncode,
        UC_Krea2TokenAttentionWeight,
        UC_Qwen3VLInputEmbeds,
        UC_VLMInputEmbeds,
    )
finally:
    cli_args.cpu = prior_cpu


def test_expression_grammar_and_nonfinite_rejection():
    value = torch.tensor([1.0, 2.0])
    assert torch.equal(encoder_helpers.evaluate_tensor_expression("clamp(a * 2, 0, 3)", {"a": value}), torch.tensor([2.0, 3.0]))
    with pytest.raises(ValueError, match="Unsupported expression element"):
        encoder_helpers.evaluate_tensor_expression("a.__class__", {"a": value})
    with pytest.raises(ValueError, match="NaN or infinite"):
        encoder_helpers.evaluate_tensor_expression("a / 0", {"a": value})


def test_embedding_output_cannot_escape_root(tmp_path):
    nested = encoder_helpers.resolve_embedding_output_path(str(tmp_path), "nested/item.safetensors")
    assert pathlib.Path(nested).is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        encoder_helpers.resolve_embedding_output_path(str(tmp_path), "../outside.safetensors")
    with pytest.raises(ValueError, match="relative"):
        encoder_helpers.resolve_embedding_output_path(str(tmp_path), str(tmp_path / "absolute.safetensors"))


def test_krea2_mapping_mirrors_core_prefix_strip():
    tokens = [
        (np.int64(151644), 1.0), (8948, 1.0), (198, 1.0), (42, 1.0), (151645, 1.0),
        (np.int64(151644), 1.0), (872, 1.0), (198, 1.0), (100, 1.0), (101, 1.0), (151645, 1.0),
    ]
    conditioning = torch.zeros(1, 3, 12 * 2560)
    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)
    assert mapping[:8] == [(-1, -1)] * 8
    assert mapping[8:] == [(0, 1), (1, 2), (2, 3)]


def test_unknown_visual_expansion_is_rejected_when_length_has_no_solution():
    tokens = [({"type": "image"}, 1.0), (10, 1.0), ({"type": "image"}, 1.0)]
    conditioning = torch.zeros(1, 8, 16)
    with pytest.raises(ValueError, match="No supported tokenizer-prefix contract"):
        encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)


def test_consensus_off_returns_reference_and_fractional_weights_stay_finite():
    first = torch.tensor([[[1.0, 0.0]]])
    second = torch.tensor([[[-1.0, 0.0]]])
    off, _ = encoder_helpers.blend_text_vectors({"a": first, "b": second}, {"blend_preset": "off"})
    assert off is first
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {
            "blend_preset": "custom",
            "blend_method": "consensus",
            "consensus_type": "mean",
            "alignment_method": "index",
            "power_alpha": 1.5,
            "similarity_threshold": -1.0,
        },
    )
    assert torch.isfinite(blended).all()


def test_contextual_weighting_does_not_scale_pooled_output():
    class Clip:
        @staticmethod
        def tokenize(text, **kwargs):
            return {"fake": [[(ord(char), 1.0) for char in text]]}

        @staticmethod
        def encode_from_tokens_scheduled(tokens):
            length = len(tokens["fake"][0])
            sequence = torch.ones(1, length, 2)
            pooled = torch.full((1, 2), 7.0)
            return [[sequence, {"pooled_output": pooled}]]

    conditioning = encoder_helpers.encode_embedding_classical_scaled_bias(Clip(), "(ab:2)c")
    sequence, metadata = conditioning[0]
    assert torch.equal(sequence[0, :2], torch.full((2, 2), 2.0))
    assert torch.equal(sequence[0, 2:], torch.ones(1, 2))
    assert torch.equal(metadata["pooled_output"], torch.full((1, 2), 7.0))


def test_contextual_weight_syntax_clean_text_matches_encoder_input():
    assert encoder_helpers.strip_contextual_weight_syntax("a (painting:-1) and ((light:2):0.5)") == "a painting and light"


def test_numbered_image_placeholders_preserve_prompt_order_and_strip_invalid(caplog):
    prompt, numbers = encoder_helpers.prepare_image_placeholder_prompt(
        "first image_input_2 then IMAGE_INPUT_1 repeat image_input_2 missing image_input_3 image_input_fusion",
        image_count=2,
        fusion_active=False,
        context="test",
    )

    assert numbers == (2, 1, 2)
    assert prompt.count(encoder_helpers.VISION_BLOCK) == 3
    assert "image_input_" not in prompt.lower()
    assert "stripped unavailable or fusion-only" in caplog.text


def test_fusion_placeholder_uses_one_slot_and_strips_the_rest(caplog):
    prompt, numbers = encoder_helpers.prepare_image_placeholder_prompt(
        "ignored image_input_1 chosen image_input_fusion removed image_input_2",
        image_count=2,
        fusion_active=True,
        context="test",
    )

    assert numbers == ()
    assert prompt.count(encoder_helpers.VISION_BLOCK) == 1
    assert "image_input_" not in prompt.lower()
    assert "stripped 2 additional" in caplog.text


def test_fusion_placeholder_accepts_image_one_alias_and_logs_fallback(caplog):
    prompt, _ = encoder_helpers.prepare_image_placeholder_prompt(
        "near image_input_1 subject",
        image_count=3,
        fusion_active=True,
        context="test",
    )

    assert prompt == f"near {encoder_helpers.VISION_BLOCK} subject"
    assert "treating image_input_1 as image_input_fusion" in caplog.text


def test_canonical_and_compatibility_schema_flags():
    assert UC_AttentionBiasTextEncode.define_schema().is_experimental
    assert UC_Krea2TokenAttentionWeight.define_schema().is_experimental
    assert TextEncodeKrea2SysEditScaledAdvAttn.define_schema().is_deprecated
    assert UC_Qwen3VLInputEmbeds.define_schema().is_deprecated
    assert not UC_VLMInputEmbeds.define_schema().is_deprecated
