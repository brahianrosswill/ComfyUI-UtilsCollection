import pathlib
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_visual_consensus_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_visual_consensus_test import encoder_helpers
    from utils_collection_visual_consensus_test.encoder_nodes import (
        AdvancedConsensusConfig,
        AdvancedVisualConfig,
        UC_AdvancedConsensusConfiguration,
        UC_AdvancedVisConEncoder,
        UC_AdvancedVisualConfiguration,
        UC_VisualConsensusConfiguration,
        VisualConsensusConfig,
    )
finally:
    cli_args.cpu = prior_cpu


def _inputs(node):
    return {value.id: value for value in node.define_schema().inputs}


def test_fresh_node_ids_names_and_socket_types():
    expected = {
        UC_AdvancedVisConEncoder: (
            "UC_AdvancedVisConEncoder",
            "Advanced Visual Consensus Encoder",
        ),
        UC_VisualConsensusConfiguration: (
            "UC_VisualConsensusConfiguration",
            "Visual Consensus Configuration",
        ),
        UC_AdvancedVisualConfiguration: (
            "UC_AdvancedVisualConfiguration",
            "Advanced Visual Configuration",
        ),
        UC_AdvancedConsensusConfiguration: (
            "UC_AdvancedConsensusConfiguration",
            "Advanced Consensus Configuration",
        ),
    }
    for node, (node_id, display_name) in expected.items():
        schema = node.define_schema()
        assert schema.node_id == node_id
        assert schema.display_name == display_name

    assert AdvancedVisualConfig.io_type == "ADVANCED_VISUAL_CONFIG"
    assert AdvancedConsensusConfig.io_type == "ADVANCED_CONSENSUS_CONFIG"
    assert VisualConsensusConfig.io_type == "VISUAL_CONSENSUS_CONFIG"
    assert _inputs(UC_AdvancedVisConEncoder)["visual_consensus_config"].optional is False


def test_joint_schema_has_literal_simple_contract():
    inputs = _inputs(UC_VisualConsensusConfiguration)
    assert list(inputs) == [
        "enable_spatial_fusion",
        "visual_fusion_method",
        "block_size",
        "dither_ratio",
        "dither_pattern",
        "seed",
        "enable_consensus",
        "consensus_preset",
        "global_scale",
        "resolution_samples",
        "advanced_visual_config",
        "advanced_consensus_config",
    ]
    assert inputs["visual_fusion_method"].options == [
        "linear",
        "checkerboard",
        "block-interleave",
        "random-dither",
    ]
    assert "off" not in inputs["consensus_preset"].options
    assert "custom" not in inputs["consensus_preset"].options


def test_advanced_configs_override_only_their_side():
    advanced_visual = UC_AdvancedVisualConfiguration.execute(
        7,
        0.2,
        "block-interleave",
        99,
        "legacy-flat",
        True,
        0.4,
        True,
        "fresh.safetensors",
    ).args[0]
    advanced_consensus = UC_AdvancedConsensusConfiguration.execute(
        "custom",
        "linear",
        "mean",
        "index",
        0.2,
        -0.1,
        1.2,
        0.7,
        False,
        1.3,
        True,
        True,
        0.8,
        True,
        7,
    ).args[0]
    config = UC_VisualConsensusConfiguration.execute(
        False,
        "checkerboard",
        2,
        0.5,
        "checkerboard",
        0,
        False,
        "baseline",
        0.5,
        5,
        advanced_visual,
        advanced_consensus,
    ).args[0]

    assert config["enable_spatial_fusion"] is False
    assert config["enable_consensus"] is False
    assert config["visual"]["visual_fusion_method"] == "spatial-checkerboard"
    assert config["visual"]["visual_block_size"] == 7
    assert config["visual"]["dither_secondary_pattern"] == "block-interleave"
    assert config["consensus"] == advanced_consensus
    assert config["consensus"]["blend_preset"] == "custom"


def test_advanced_consensus_preset_is_authoritative_and_defaults_to_custom():
    inputs = _inputs(UC_AdvancedConsensusConfiguration)
    preset = inputs["consensus_preset"]
    assert preset.default == "custom"
    assert preset.options[0] == "custom"
    assert "baseline" in preset.options
    assert "off" not in preset.options

    config = UC_AdvancedConsensusConfiguration.execute(
        "high_clarity",
        "linear",
        "mean",
        "index",
        0.1,
        -0.2,
        1.0,
        0.5,
        False,
        1.4,
        True,
        True,
        0.65,
        True,
        5,
    ).args[0]
    assert config["blend_preset"] == "high_clarity"
    assert config["position_weight"] == 0.65
    assert config["preserve_common_prefix"] is True
    assert config["global_scale"] == 1.4
    assert config["resolution_samples"] == 5
    resolved = encoder_helpers.resolve_consensus_blend_settings(config)
    assert resolved["blend_preset"] == "high_clarity"
    assert resolved["power_alpha"] == 3.0
    assert resolved["position_weight"] == 0.65
    assert resolved["preserve_common_prefix"] is True
    assert resolved["global_scale"] == 1.4


def test_simple_resolution_samples_reaches_consensus_config():
    config = UC_VisualConsensusConfiguration.execute(
        True,
        "linear",
        2,
        0.5,
        "checkerboard",
        0,
        True,
        "baseline",
        1.0,
        7,
    ).args[0]

    assert config["consensus"]["resolution_samples"] == 7


def test_one_batched_socket_is_one_lane_of_separate_sources():
    images = torch.arange(3 * 2 * 2 * 1).reshape(3, 2, 2, 1)
    lanes, references = encoder_helpers.build_visual_consensus_batch_lanes(
        {"image0": images}
    )
    assert len(lanes) == 1
    assert len(lanes[0]) == 3
    assert len(references) == 3
    assert all(image.shape == (1, 2, 2, 1) for image in lanes[0])
    separate_lanes, _ = encoder_helpers.build_visual_consensus_batch_lanes(
        {
            "image0": images[0:1],
            "image1": images[1:2],
            "image2": images[2:3],
        }
    )
    assert len(separate_lanes) == 1
    assert all(
        torch.equal(batched, separate)
        for batched, separate in zip(lanes[0], separate_lanes[0])
    )


def test_multiple_batches_pair_by_index_and_broadcast_singletons():
    first = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    second = torch.tensor([[[[10.0]]], [[[20.0]]], [[[30.0]]]])
    singleton = torch.tensor([[[[99.0]]]])
    lanes, _ = encoder_helpers.build_visual_consensus_batch_lanes(
        {"image0": first, "image1": singleton, "image2": second}
    )
    assert [
        [float(source.item()) for source in lane]
        for lane in lanes
    ] == [
        [1.0, 99.0, 10.0],
        [2.0, 99.0, 20.0],
        [3.0, 99.0, 30.0],
    ]


def test_multiple_unequal_batches_fail_concisely():
    with pytest.raises(ValueError, match="batch lengths must match or be 1"):
        encoder_helpers.build_visual_consensus_batch_lanes(
            {
                "image0": torch.zeros(2, 1, 1, 1),
                "image1": torch.zeros(3, 1, 1, 1),
            }
        )


def test_complete_conditioning_helper_matches_consensus_math(monkeypatch):
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "get_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "intermediate_dtype",
        lambda: torch.float32,
    )
    first = [[torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), {"pooled_output": torch.tensor([[1.0, 0.0]])}]]
    second = [[torch.tensor([[[3.0, 0.0], [0.0, 3.0]]]), {"pooled_output": torch.tensor([[3.0, 0.0]])}]]
    config = {"blend_preset": "custom", "blend_method": "linear", "global_scale": 1.0}
    actual = encoder_helpers.blend_complete_conditionings([first, second], config)
    expected_tensor, expected_pooled = encoder_helpers.blend_text_vectors(
        {"a": first[0][0], "b": second[0][0]},
        config,
        {"a": first[0][1]["pooled_output"], "b": second[0][1]["pooled_output"]},
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
    )
    assert torch.equal(actual[0][0], expected_tensor)
    assert torch.equal(actual[0][1]["pooled_output"], expected_pooled)


def _execution_config(spatial=True, consensus=True, samples=1):
    return {
        "enable_spatial_fusion": spatial,
        "enable_consensus": consensus,
        "visual": {
            "visual_fusion_method": "spatial-checkerboard",
            "visual_encoder_path": "grid-deepstack",
        },
        "consensus": {
            "blend_preset": "baseline",
            "resolution_samples": samples,
        },
    }


def _mock_execution_boundaries(monkeypatch):
    encoded = []
    fused = []
    blended = []

    def encode(clip, source, resolution, prompt, path):
        encoded.append((float(source.flatten()[0]), resolution, path))
        tensor = torch.tensor([[[float(source.flatten()[0])]]])
        conditioning = [[tensor, {}]]
        return {"conditioning": conditioning}

    def spatial(branches, config, clip, allow_export):
        fused.append((len(branches), allow_export))
        return branches[0]["conditioning"]

    def consensus(conditionings, config):
        blended.append((len(conditionings), config))
        return conditionings[0]

    monkeypatch.setattr(
        encoder_helpers, "_encode_visual_consensus_source", encode
    )
    monkeypatch.setattr(
        encoder_helpers, "_spatially_fuse_visual_consensus_sources", spatial
    )
    monkeypatch.setattr(
        encoder_helpers, "blend_complete_conditionings", consensus
    )
    return encoded, fused, blended


def _execute_mocked(image_inputs, config):
    return encoder_helpers.execute_advanced_visual_consensus(
        object(),
        "prompt",
        "",
        384,
        image_inputs,
        config,
        "Original",
        "off",
        None,
        1.0,
        8,
        lambda clip, conditioning, latents, mode: conditioning,
    )


def test_one_batched_socket_fuses_all_sources_at_three_resolutions(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    _execute_mocked({"image0": images}, _execution_config())

    assert len(encoded) == 9
    assert [resolution for _, resolution, _ in encoded[:3]] == [384, 384, 384]
    assert [count for count, _ in fused] == [3, 3, 3]
    assert blended[0][0] == 3


def test_three_batch_lanes_suppress_automatic_resolution_expansion(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    first = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    second = torch.tensor([[[[4.0]]], [[[5.0]]], [[[6.0]]]])
    _execute_mocked(
        {"image0": first, "image1": second},
        _execution_config(samples=1),
    )

    assert len(encoded) == 6
    assert [count for count, _ in fused] == [2, 2, 2]
    assert blended[0][0] == 3


def test_consensus_only_retains_every_source_resolution_conditioning(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    _execute_mocked(
        {"image0": images},
        _execution_config(spatial=False, consensus=True),
    )

    assert len(encoded) == 9
    assert fused == []
    assert blended[0][0] == 9


def test_configured_five_resolution_samples_reach_consensus(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    _execute_mocked(
        {"image0": torch.ones(1, 2, 2, 1)},
        _execution_config(samples=5),
    )

    assert len(encoded) == 5
    assert len(fused) == 5
    assert blended[0][0] == 5


def test_both_disabled_encodes_only_first_source_at_base_resolution(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    output = _execute_mocked(
        {"image0": images},
        _execution_config(spatial=False, consensus=False),
    )

    assert encoded == [(1.0, 384, "grid-deepstack")]
    assert fused == []
    assert blended == []
    assert output[0][0].item() == 1.0


def test_original_resolution_requires_three_lanes_for_consensus(monkeypatch):
    _mock_execution_boundaries(monkeypatch)
    with pytest.raises(ValueError, match="Original VLM resolution needs at least three"):
        encoder_helpers.execute_advanced_visual_consensus(
            object(),
            "prompt",
            "",
            0,
            {"image0": torch.ones(1, 2, 2, 1)},
            _execution_config(),
            "Original",
            "off",
            None,
            1.0,
            8,
            lambda clip, conditioning, latents, mode: conditioning,
        )


def test_consensus_disabled_preserves_execution_lanes_as_batch(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    first = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    second = torch.tensor([[[[4.0]]], [[[5.0]]], [[[6.0]]]])
    output = _execute_mocked(
        {"image0": first, "image1": second},
        _execution_config(spatial=True, consensus=False),
    )

    assert len(encoded) == 6
    assert [count for count, _ in fused] == [2, 2, 2]
    assert blended == []
    assert output[0][0].shape[0] == 3
    assert output[0][0].flatten().tolist() == [1.0, 2.0, 3.0]
