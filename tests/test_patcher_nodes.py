import pathlib
import sys
import types

import pytest
import torch

import comfy.model_patcher


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_patcher_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_patcher_test import patcher_helpers, patcher_nodes


def _cache(
    *,
    threshold=0.1,
    start=0.0,
    end=1.0,
    max_steps=2,
    device="auto",
    verbose=False,
):
    cache = patcher_helpers.MiniMaxH3Cache(
        reuse_threshold=threshold,
        start_percent=start,
        end_percent=end,
        max_steps=max_steps,
        device=device,
        verbose=verbose,
    )
    cache.begin(10)
    return cache


def _cache_args(image, timestep, cache_ranges=((0, 4),)):
    return {
        "img": image,
        "timestep": torch.tensor([timestep]),
        "cache_ranges": cache_ranges,
        "block_count": 2,
    }


def test_schema_exposes_stable_cache_controls():
    schema = patcher_nodes.UC_MiniMaxH3Cache.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_MiniMaxH3Cache"
    assert schema.is_experimental
    assert [value.id for value in schema.inputs] == [
        "model",
        "reuse_threshold",
        "start_percent",
        "end_percent",
        "max_steps",
        "device",
        "verbose",
    ]
    assert inputs["reuse_threshold"].default == 0.05
    assert inputs["start_percent"].default == 0.15
    assert inputs["end_percent"].default == 0.9
    assert inputs["max_steps"].default == 2
    assert inputs["device"].default == "auto"


def test_spectrum_schema_exposes_measurable_experimental_controls():
    schema = patcher_nodes.UC_MiniMaxH3Spectrum.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_MiniMaxH3Spectrum"
    assert schema.is_experimental
    assert inputs["execution_mode"].default == "spectrum"
    assert inputs["degree"].default == 1
    assert inputs["ridge_lambda"].default == 0.1
    assert inputs["warmup_steps"].default == 1
    assert inputs["video_blend_weight"].default == 0.5
    assert inputs["offline_smoothing_replay"].default is True


def test_spectrum_ridge_weights_reproduce_known_polynomial():
    coordinates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    forecaster = patcher_helpers.HistoryWeightForecaster(
        degree=2, ridge_lambda=0.0, max_history=5
    )
    for coordinate in coordinates:
        feature = torch.tensor([[[2.0 + 3.0 * coordinate - coordinate * coordinate]]])
        forecaster.update(coordinate, feature)

    predicted = forecaster.predict(0.25, 1.0)
    assert predicted.item() == pytest.approx(
        2.0 + 3.0 * 0.25 - 0.25**2, abs=1e-5
    )


def _spectrum_config(**overrides):
    values = {
        "enabled": True,
        "force_actual": False,
        "degree": 2,
        "ridge_lambda": 0.1,
        "window_size": 2.0,
        "flex_window": 0.75,
        "warmup_steps": 3,
        "tail_actual_steps": 1,
        "max_history": 8,
        "blend_weight": 1.0,
        "audio_blend_weight": 0.0,
        "history_storage": "system_ram",
        "bootstrap_first_forecast": True,
        "offline_smoothing_replay": False,
        "offline_archive_storage": "system_ram",
        "debug": False,
    }
    values.update(overrides)
    return patcher_helpers.SpectrumH3Config(**values)


def test_spectrum_config_rejects_insufficient_history():
    with pytest.raises(ValueError, match="max_history"):
        _spectrum_config(degree=4, max_history=4, bootstrap_first_forecast=False).validate()


def test_spectrum_forced_actual_disables_replay(monkeypatch):
    captured = {}

    def patch(model, config):
        captured["config"] = config
        return model

    monkeypatch.setattr(patcher_nodes, "patch_minimax_h3_spectrum_model", patch)
    model = object()
    patcher_nodes.UC_MiniMaxH3Spectrum.execute(
        model,
        "forced_actual",
        1,
        0.1,
        2.0,
        0.75,
        1,
        1,
        8,
        0.5,
        0.0,
        "system_ram",
        True,
        True,
        "system_ram",
        False,
    )

    assert captured["config"].force_actual is True
    assert captured["config"].offline_smoothing_replay is False


def test_spectrum_block_loop_observes_named_audio_then_video_targets(monkeypatch):
    class FakeRuntime:
        offline_phase = None

        def begin_model_call(self, *_args, **kwargs):
            assert kwargs["expected_shape"] == (1, 3, 2)
            assert dict(kwargs["topology"])["target_audio_rows"] == 1
            assert dict(kwargs["topology"])["target_video_rows"] == 2
            return 0, True

        def observe_actual(self, _run, _step, _call, feature):
            self.feature = feature

    monkeypatch.setattr(patcher_helpers, "SpectrumH3Runtime", FakeRuntime)
    runtime = FakeRuntime()
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    options = {
        patcher_helpers.RUNTIME_KEY: runtime,
        patcher_helpers.RUN_ID_KEY: 1,
        patcher_helpers.STEP_ID_KEY: 2,
    }
    output = patcher_helpers.SpectrumH3BlockLoop()(
        {
            "img": hidden,
            "transformer_options": options,
            "target_ranges": ((4, 6, "video"), (2, 3, "audio")),
            "block_count": 2,
        },
        {"original_block": lambda args: {"img": args["img"] + 10}},
    )

    assert torch.equal(output["img"], hidden + 10)
    assert torch.equal(runtime.feature, torch.cat(((hidden + 10)[2:3], (hidden + 10)[4:6])).unsqueeze(0))


def test_spectrum_block_loop_inserts_forecast_into_named_targets(monkeypatch):
    class FakeRuntime:
        offline_phase = None

        def begin_model_call(self, *_args, **_kwargs):
            return 0, False

        def predict(self, *_args, **_kwargs):
            return torch.tensor([[[20.0], [40.0], [50.0]]])

    monkeypatch.setattr(patcher_helpers, "SpectrumH3Runtime", FakeRuntime)
    runtime = FakeRuntime()
    hidden = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    result = patcher_helpers.SpectrumH3BlockLoop()(
        {
            "img": hidden,
            "transformer_options": {
                patcher_helpers.RUNTIME_KEY: runtime,
                patcher_helpers.RUN_ID_KEY: 1,
                patcher_helpers.STEP_ID_KEY: 2,
            },
            "target_ranges": ((4, 6, "video"), (2, 3, "audio")),
            "block_count": 2,
        },
        {"original_block": lambda _args: pytest.fail("forecast ran transformer")},
    )["img"]

    assert result[:, 0].tolist() == [0.0, 1.0, 20.0, 3.0, 40.0, 50.0]


def test_cache_reuses_residual_then_honors_maximum_skip_count():
    cache = _cache(max_steps=1)
    calls = []

    def original(args):
        calls.append(args["img"].clone())
        return {"img": args["img"] + 2.0}

    first = torch.ones((4, 8))
    assert torch.equal(
        cache(_cache_args(first, 1000.0), {"original_block": original})["img"],
        first + 2.0,
    )

    second = first + 0.001
    skipped = cache(_cache_args(second, 900.0), {"original_block": original})["img"]
    assert torch.allclose(skipped, second + 2.0)
    assert len(calls) == 1

    third = first + 0.002
    cache(_cache_args(third, 800.0), {"original_block": original})
    assert len(calls) == 2


def test_cache_recomputes_for_threshold_range_and_layout_changes():
    calls = []

    def original(args):
        calls.append(args["img"].clone())
        return {"img": args["img"] + 1.0}

    cache = _cache(threshold=0.01)
    cache(_cache_args(torch.ones((4, 8)), 1000.0), {"original_block": original})
    cache(_cache_args(torch.full((4, 8), 2.0), 900.0), {"original_block": original})
    assert len(calls) == 2

    range_cache = _cache(start=0.5)
    range_cache(
        _cache_args(torch.ones((4, 8)), 1000.0), {"original_block": original}
    )
    range_cache(
        _cache_args(torch.ones((4, 8)), 900.0), {"original_block": original}
    )
    assert len(calls) == 4

    layout_cache = _cache()
    layout_cache(
        _cache_args(torch.ones((4, 8)), 1000.0), {"original_block": original}
    )
    layout_cache(
        _cache_args(torch.ones((4, 8)), 900.0, ((1, 4),)),
        {"original_block": original},
    )
    assert len(calls) == 6


def test_cpu_cache_preserves_output_shape_and_dtype():
    cache = _cache(device="cpu")

    def original(args):
        return {"img": args["img"] + 0.5}

    image = torch.ones((4, 8), dtype=torch.float32)
    cache(_cache_args(image, 1000.0), {"original_block": original})
    output = cache(
        _cache_args(image + 0.001, 900.0), {"original_block": original}
    )["img"]

    assert cache.cached_residual.device.type == "cpu"
    assert output.shape == image.shape
    assert output.dtype == image.dtype


def test_sampling_scope_always_clears_cache_state():
    cache = _cache()
    scope = patcher_helpers.MiniMaxH3SamplingScope(cache)

    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    sigmas = torch.tensor([1.0, 0.5, 0.0])
    with pytest.raises(RuntimeError, match="sampling failed"):
        scope(fail, None, None, None, sigmas)

    assert cache.cached_residual is None
    assert cache.step_counter == 0


def test_block_runner_preserves_double_block_replacements(monkeypatch):
    prefetch_events = []
    monkeypatch.setattr(
        patcher_helpers.comfy.model_prefetch,
        "make_prefetch_queue",
        lambda blocks, device, options: "queue",
    )
    monkeypatch.setattr(
        patcher_helpers.comfy.model_prefetch,
        "prefetch_queue_pop",
        lambda queue, device, block: prefetch_events.append(block),
    )

    class Block:
        def __call__(self, image, *args, **kwargs):
            return image + 1.0

    blocks = [Block(), Block()]
    model = types.SimpleNamespace(blocks=blocks)

    def replacement(args, extra_options):
        result = extra_options["original_block"](args)
        return {"img": result["img"] + 10.0}

    output = patcher_helpers.run_minimax_h3_blocks(
        model,
        torch.zeros((2, 4)),
        torch.zeros((1, 4)),
        [],
        torch.zeros((1, 4)),
        {"patches_replace": {"dit": {("double_block", 0): replacement}}},
    )

    assert torch.equal(output, torch.full((2, 4), 12.0))
    assert prefetch_events == [blocks[0], blocks[1], None]


def test_cached_forward_matches_current_core_audio_output_contract(monkeypatch):
    layout = types.SimpleNamespace(
        signature=(1, 1, 2, 2, 1),
        segments=[(0, 1, "text"), (1, 3, "audio"), (3, 4, "video")],
        img_update=torch.ones(1, dtype=torch.bool),
        audio_update=torch.ones(2, dtype=torch.bool),
        seq_len=4,
        position_ids=torch.zeros((4, 3)),
    )
    video_result = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    audio_result = torch.arange(1.0, 9.0).reshape(2, 4)
    model = types.SimpleNamespace(
        patch_size=(1, 2, 2),
        sigma_shift_video=1.0,
        sigma_shift_audio=1.0,
        hidden_size=4,
        use_adaln_curves=False,
        blocks=[],
        latents_dim=1,
        video_patch_proj=lambda rows: rows,
        audio_patch_proj=lambda rows: rows,
        _cond_video_rows=lambda payload, device: None,
        _cond_audio_rows=lambda payload, device: None,
        time_embedder=lambda values: values[:, None].expand(-1, 4),
        rope_freqs=lambda position_ids, device: torch.zeros((1, 1)),
        final_layer=lambda hidden, t_emb, video_seg, audio_seg: (
            video_result,
            audio_result,
        ),
    )
    monkeypatch.setattr(
        patcher_helpers.minimax_model,
        "rope_rotation_table",
        lambda frequencies, dtype: frequencies,
    )

    video = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float16)
    audio = torch.zeros((1, 4, 2, 1), dtype=torch.float16)
    context = torch.zeros((1, 1, 4), dtype=torch.float32)

    assert not hasattr(patcher_helpers.minimax_model, "time_shift_slope")
    output = patcher_helpers.minimax_h3_block_patch_forward(
        model,
        [video, audio],
        torch.tensor([500.0]),
        context,
        minimax_payload={"layout": layout},
    )

    expected_video = -patcher_helpers.minimax_model.unpatchify_video(
        video_result, 1, 1, 1, 1, (1, 2, 2)
    ).to(video.dtype)
    expected_audio = -patcher_helpers.minimax_model.unpack_audio(audio_result).to(
        audio.dtype
    )
    assert torch.equal(output[0], expected_video)
    assert torch.equal(output[1], expected_audio)


def test_model_helper_adds_only_reversible_instance_patch(monkeypatch):
    class FakeH3:
        def _forward(self):
            return "original"

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)
    diffusion_model = FakeH3()

    class FakePatcher:
        def __init__(self, diffusion):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.object_patches = {}
            self.replacements = []
            self.wrappers = []

        def clone(self):
            return FakePatcher(self.model.diffusion_model)

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def set_model_patch_replace(self, *args):
            self.replacements.append(args)

        def add_wrapper(self, *args):
            self.wrappers.append(args)

    original = FakePatcher(diffusion_model)
    original_class_forward = FakeH3.__dict__["_forward"]
    patched = patcher_helpers.patch_minimax_h3_cache_model(
        original, 0.1, 0.15, 0.9, 2, "auto", False
    )

    assert original.object_patches == {}
    assert set(patched.object_patches) == {"diffusion_model._forward"}
    assert patched.object_patches["diffusion_model._forward"].__self__ is diffusion_model
    assert patched.replacements[0][1:] == ("dit", "block_loop", 0)
    assert patched.wrappers[0][0] == patcher_helpers.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
    assert FakeH3.__dict__["_forward"] is original_class_forward


def test_spectrum_helper_uses_clone_scoped_object_patch_and_hybrid_boundary(monkeypatch):
    class FakeH3:
        def _forward(self):
            return "original"

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)
    monkeypatch.setattr(patcher_helpers, "install_sampler_wrappers", lambda model, runtime: model.wrappers.append(runtime))

    class FakePatcher:
        def __init__(self, diffusion):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.model_options = {}
            self.object_patches = {}
            self.replacements = []
            self.wrappers = []

        def clone(self):
            return FakePatcher(self.model.diffusion_model)

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def set_model_patch_replace(self, *args):
            self.replacements.append(args)

    original = FakePatcher(FakeH3())
    original_class_forward = FakeH3.__dict__["_forward"]
    patched = patcher_helpers.patch_minimax_h3_spectrum_model(
        original, _spectrum_config(degree=1, warmup_steps=1)
    )

    assert original.model_options == {}
    assert patched.model_options[patcher_helpers.MINIMAX_H3_SPECTRUM_OWNER_KEY] is True
    assert set(patched.object_patches) == {"diffusion_model._forward"}
    assert patched.object_patches["diffusion_model._forward"].__self__ is original.model.diffusion_model
    assert isinstance(patched.replacements[0][0], patcher_helpers.SpectrumH3BlockLoop)
    assert patched.replacements[0][1:] == ("dit", "block_loop", 0)
    assert len(patched.wrappers) == 1
    assert FakeH3.__dict__["_forward"] is original_class_forward


def test_cache_and_spectrum_reject_both_stacking_orders():
    cache_model = types.SimpleNamespace(
        model_options={patcher_helpers.MINIMAX_H3_SPECTRUM_OWNER_KEY: True}
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        patcher_helpers.patch_minimax_h3_cache_model(
            cache_model, 0.1, 0.1, 0.9, 2, "auto", False
        )

    spectrum_model = types.SimpleNamespace(
        model_options={patcher_helpers.MINIMAX_H3_CACHE_OWNER_KEY: True}
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        patcher_helpers.patch_minimax_h3_spectrum_model(
            spectrum_model, _spectrum_config(degree=1, warmup_steps=1)
        )


def test_model_helper_rejects_invalid_inputs(monkeypatch):
    class FakeH3:
        pass

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)

    class FakePatcher:
        def __init__(self):
            self.model = types.SimpleNamespace(diffusion_model=object())

        def clone(self):
            return self

    with pytest.raises(ValueError, match="start percent"):
        patcher_helpers.patch_minimax_h3_cache_model(
            FakePatcher(), 0.1, 0.9, 0.1, 2, "auto", False
        )
    with pytest.raises(ValueError, match="requires a MiniMax H3"):
        patcher_helpers.patch_minimax_h3_cache_model(
            FakePatcher(), 0.1, 0.1, 0.9, 2, "auto", False
        )


def test_core_object_patch_restores_original_bound_method():
    class Diffusion:
        def _forward(self):
            return "original"

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.diffusion_model = Diffusion()

    tiny_model = TinyModel()
    patcher = comfy.model_patcher.ModelPatcher(
        tiny_model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=1,
    )

    def replacement(self):
        return "patched"

    patcher.add_object_patch(
        "diffusion_model._forward",
        types.MethodType(replacement, tiny_model.diffusion_model),
    )
    patcher.patch_model(load_weights=False)
    assert tiny_model.diffusion_model._forward() == "patched"

    patcher.unpatch_model(unpatch_weights=False)
    assert tiny_model.diffusion_model._forward() == "original"
