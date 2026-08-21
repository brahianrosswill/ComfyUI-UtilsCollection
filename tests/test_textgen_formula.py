import ast
import inspect
import pathlib
import subprocess
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_textgen_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_textgen_test import textgen_nodes
finally:
    cli_args.cpu = prior_cpu


def _images():
    return {
        "a": torch.tensor([[[[0.2, 0.5, 0.8]]]], dtype=torch.float32),
        "b": torch.tensor([[[[0.6, 0.4, 0.1]]]], dtype=torch.float32),
        "image_input_1": torch.tensor([[[[0.2, 0.5, 0.8]]]], dtype=torch.float32),
    }


def test_textgen_formula_preserves_arithmetic_aliases_and_clamping():
    images = _images()
    arithmetic = textgen_nodes.evaluate_formula("-a + (b * 2) ** 1 / 2", images)
    named = textgen_nodes.evaluate_formula("(image_input_1 + b) / 2", images)

    assert arithmetic.dtype == images["a"].dtype
    assert arithmetic.shape == images["a"].shape
    assert torch.allclose(arithmetic, torch.clamp(-images["a"] + images["b"], 0.0, 1.0))
    assert torch.allclose(named, (images["a"] + images["b"]) / 2)
    assert textgen_nodes.evaluate_formula("a * 10", images).max().item() == 1.0


def test_textgen_formula_preserves_named_tensor_functions():
    images = _images()
    result = textgen_nodes.evaluate_formula("clamp(max(abs(a - b), min(a, b)), 0.1, 0.6)", images)
    expected = torch.clamp(
        torch.maximum(torch.abs(images["a"] - images["b"]), torch.minimum(images["a"], images["b"])),
        0.1,
        0.6,
    )
    assert torch.allclose(result, expected)


@pytest.mark.parametrize("expression", [
    "unknown + a",
    "a.__class__",
    "a.mean()",
    "a[0]",
    "sum([a])",
    "clamp(a, min=0.0, max=1.0)",
    "a if True else b",
])
def test_textgen_formula_rejects_unsupported_expression_elements(expression):
    with pytest.raises(RuntimeError, match="Error evaluating textgen visual math expression"):
        textgen_nodes.evaluate_formula(expression, _images())


@pytest.mark.parametrize("expression", ["a +", "a / 0"])
def test_textgen_formula_rejects_invalid_or_nonfinite_results(expression):
    with pytest.raises(RuntimeError, match="Error evaluating textgen visual math expression"):
        textgen_nodes.evaluate_formula(expression, _images())


def test_global_and_inline_formula_routes_evaluate_their_image_expressions(monkeypatch):
    class Clip:
        tokenizer = types.SimpleNamespace(clip_name="qwen3vl_4b")

        @staticmethod
        def tokenize(_prompt, **_kwargs):
            return {"qwen": [[(1, 1.0)]]}

        @staticmethod
        def generate(_tokens, **_kwargs):
            return [7]

        @staticmethod
        def decode(_tokens, **_kwargs):
            return "decoded"

    calls = []
    real_evaluate = textgen_nodes.evaluate_formula

    def record(expression, images):
        calls.append((expression, set(images)))
        return real_evaluate(expression, images)

    monkeypatch.setattr(textgen_nodes, "evaluate_formula", record)
    images = {
        "image_0": torch.full((1, 2, 2, 3), 0.2),
        "image_1": torch.full((1, 2, 2, 3), 0.3),
    }

    global_result = textgen_nodes.UC_TextGenerate.execute(
        Clip(),
        "describe",
        "",
        0,
        12,
        {"sampling_mode": "off"},
        formula="a + b",
        image_inputs=images,
    )
    inline_result = textgen_nodes.UC_TextGenerate.execute(
        Clip(),
        "compare |a + b|",
        "",
        0,
        12,
        {"sampling_mode": "off"},
        image_inputs=images,
    )

    assert global_result.args == inline_result.args == ("decoded", 0)
    assert calls == [
        ("a + b", {"image_input_1", "image_input_2", "a", "b"}),
        ("a + b", {"image_input_1", "image_input_2", "a", "b"}),
    ]


def test_text_generate_schema_has_no_obsolete_blend_config():
    input_ids = [value.id for value in textgen_nodes.UC_TextGenerate.define_schema().inputs]
    assert "blend_config" not in input_ids
    assert "model_type" not in input_ids
    assert "image_inputs" in input_ids
    assert "blend_config" not in inspect.signature(textgen_nodes.UC_TextGenerate.execute).parameters
    assert "model_type" not in inspect.signature(textgen_nodes.UC_TextGenerate.execute).parameters
    assert not hasattr(textgen_nodes, "BlendConfig")
    assert not hasattr(textgen_nodes, "evaluate_image_consensus_blend")


def test_text_generate_schema_uses_shared_numeric_vlm_resolution_contract():
    resolution = {
        value.id: value for value in textgen_nodes.UC_TextGenerate.define_schema().inputs
    }["vlm_resolution"]

    assert resolution.io_type == "INT"
    assert resolution.default == 384
    assert resolution.min == 0
    assert resolution.max == 4096
    assert resolution.step == 32


def test_text_generate_appends_optional_visual_fusion_config():
    inputs = textgen_nodes.UC_TextGenerate.define_schema().inputs
    assert "visual_fusion_config" in {value.id for value in inputs}
    assert "visual_fusion_config" in inspect.signature(textgen_nodes.UC_TextGenerate.execute).parameters


def test_text_generate_schema_exposes_token_fusion_capability():
    schema = textgen_nodes.UC_TextGenerate.define_schema()
    assert schema.display_name == "Text Generate (TokenFusion)"
    assert "visual_fusion_config" in [value.id for value in schema.inputs]
    assert [output.id for output in textgen_nodes.UC_TextGenerate.define_schema().outputs] == [
        "generated_text",
        "seed",
    ]


def test_text_generate_parenthesis_escaping_is_optional_and_final():
    clip = _GenerateClip()
    clip.decode = lambda *args, **kwargs: "plain (Overwatch), (banana)"
    common = (clip, "hello", "", 0, 12, {"sampling_mode": "off"})

    assert textgen_nodes.UC_TextGenerate.execute(*common).args == ("plain (Overwatch), (banana)", 0)
    assert textgen_nodes.UC_TextGenerate.execute(*common, escape_parentheses=True).args == (
        r"plain \(Overwatch\), \(banana\)",
        0,
    )


class _GenerateClip:
    def __init__(self, family="qwen3vl_4b"):
        self.tokenizer = types.SimpleNamespace(clip_name=family)
        self.tokenize_calls = []
        self.generate_calls = []

    def tokenize(self, prompt, **kwargs):
        self.tokenize_calls.append((prompt, kwargs))
        return {"qwen": [[(1, 1.0)]]}

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((tokens, kwargs))
        return [7]

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded"


@pytest.mark.parametrize("resolution", [384, 0])
def test_text_generate_delegates_image_sizing_to_shared_vlm_helper(monkeypatch, resolution):
    calls = []
    image = torch.zeros((1, 32, 64, 3))

    def prepare(source, selected_resolution):
        calls.append((source, selected_resolution))
        return source

    monkeypatch.setattr(textgen_nodes, "prepare_vlm_image", prepare)
    textgen_nodes.UC_TextGenerate.execute(
        _GenerateClip(),
        "describe image_input_1",
        "",
        resolution,
        12,
        {"sampling_mode": "off"},
        image_inputs={"image1": image},
    )

    assert [source is image for source, _ in calls] == [True, True]
    assert [selected_resolution for _, selected_resolution in calls] == [resolution, resolution]


@pytest.mark.parametrize("config", [None, {"visual_fusion_method": "off"}])
def test_text_generate_disconnected_or_off_uses_original_generation_path(config):
    clip = _GenerateClip()
    result = textgen_nodes.UC_TextGenerate.execute(
        clip, "hello", "", 0, 12, {"sampling_mode": "off"},
        image_inputs={}, visual_fusion_config=config,
    )
    assert result.args == ("decoded", 0)
    assert len(clip.tokenize_calls) == len(clip.generate_calls) == 1
    assert clip.generate_calls[0][1]["max_length"] == 12


def test_text_generate_retries_blank_sampled_outputs_and_returns_successful_seed():
    clip = _GenerateClip()
    responses = iter(["", " \n", "caption"])
    clip.decode = lambda *args, **kwargs: next(responses)
    sampling = {
        "sampling_mode": "on",
        "seed": 41,
        "empty_response_retries": 4,
    }

    result = textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "", 0, 12, sampling,
        image_inputs={}, visual_fusion_config=None,
    )

    assert result.args == ("caption", 43)
    assert [call[1]["seed"] for call in clip.generate_calls] == [41, 42, 43]


def test_text_generate_retry_seed_wraps_at_unsigned_64_bit_limit():
    clip = _GenerateClip()
    responses = iter(["", "caption"])
    clip.decode = lambda *args, **kwargs: next(responses)
    sampling = {
        "sampling_mode": "on",
        "seed": 0xffffffffffffffff,
        "empty_response_retries": 1,
    }

    result = textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "", 0, 12, sampling,
        image_inputs={}, visual_fusion_config=None,
    )

    assert result.args == ("caption", 0)
    assert [call[1]["seed"] for call in clip.generate_calls] == [
        0xffffffffffffffff,
        0,
    ]


def test_text_generate_does_not_retry_deterministic_blank_output():
    clip = _GenerateClip()
    clip.decode = lambda *args, **kwargs: ""

    result = textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "", 0, 12,
        {"sampling_mode": "off", "empty_response_retries": 4},
        image_inputs={}, visual_fusion_config=None,
    )

    assert result.args == ("", 0)
    assert len(clip.generate_calls) == 1


def test_qwen3vl_system_prompt_uses_official_assistant_generation_boundary():
    clip = _GenerateClip()
    textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "caption rules", 0, 12, {"sampling_mode": "off"},
        image_inputs={}, visual_fusion_config=None,
    )

    prompt = clip.tokenize_calls[0][0]
    assert prompt == (
        "<|im_start|>system\ncaption rules<|im_end|>\n"
        "<|im_start|>user\ndescribe<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert "<think>" not in prompt


def test_qwen3vl_fused_system_prompt_uses_official_assistant_generation_boundary(monkeypatch):
    captured = {}

    def generate(clip, full_prompt, images, config, generation_args, thinking=False):
        captured["prompt"] = full_prompt
        captured["images"] = images
        return [7]

    monkeypatch.setattr(textgen_nodes, "generate_fused_qwen3vl", generate)
    clip = _GenerateClip()
    image_a = torch.zeros((1, 2, 2, 3))
    image_b = torch.ones((1, 2, 2, 3))
    textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "caption rules", 0, 12, {"sampling_mode": "off"},
        formula="a",
        image_inputs={"image1": image_a, "image2": image_b},
        visual_fusion_config={"visual_fusion_method": "linear"},
    )

    assert captured["prompt"].endswith("<|im_start|>assistant\n")
    assert "<think>" not in captured["prompt"]
    assert len(captured["images"]) == 2
    assert torch.equal(captured["images"][0], image_a)
    assert torch.equal(captured["images"][1], image_b)


def test_active_visual_fusion_rejects_unsupported_model_without_tokenizing():
    clip = _GenerateClip("gemma3_12b")
    image = torch.zeros((1, 2, 3, 3))
    with pytest.raises(ValueError, match="only by Core Qwen3-VL and Qwen3.5"):
        textgen_nodes.UC_TextGenerate.execute(
            clip, "describe", "", 0, 12, {"sampling_mode": "off"},
            image_inputs={"image1": image},
            visual_fusion_config={"visual_fusion_method": "linear"},
        )
    assert clip.tokenize_calls == []


def test_qwen35_fused_generation_uses_primary_visual_block_and_mrope():
    class Transformer:
        generated = None

        def generate(self, embeds, **kwargs):
            self.generated = (embeds, kwargs)
            return [9]

    class Model:
        def __init__(self):
            self.transformer = Transformer()
            self.execution_device = torch.device("cpu")

        def reset_clip_options(self):
            pass

        def set_clip_options(self, options):
            self.execution_device = options["execution_device"]

        def process_tokens(self, rows, device):
            image = next(value for value in rows[0] if isinstance(value, dict))
            source = float(image["data"].flatten()[0])
            embeds = torch.zeros((1, 6, 2), device=device)
            embeds[0, 1:5] = source
            info = [{"type": "image", "index": 1, "size": 4, "extra": torch.tensor([[1, 4, 4]])}]
            return embeds, None, None, info

    model = Model()
    stage = types.SimpleNamespace(clip="qwen35_2b", qwen35_2b=model)

    class Clip:
        cond_stage_model = stage
        patcher = types.SimpleNamespace(load_device=torch.device("cpu"))

        @staticmethod
        def load_model():
            pass

        @staticmethod
        def tokenize(prompt, images, **kwargs):
            return {"qwen35": [[(10, 1.0), ({"type": "image", "data": images[0]}, 1.0), (11, 1.0)]]}

    images = [torch.zeros((1, 2, 2, 3)), torch.ones((1, 2, 2, 3))]
    args = {"do_sample": False, "max_length": 4, "temperature": 1.0, "top_k": 50,
            "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "seed": None}
    result = textgen_nodes.generate_fused_qwen35(
        Clip(), "prompt", images, {"visual_fusion_method": "spatial-checkerboard"}, args
    )

    assert result == [9]
    embeds, forwarded = model.transformer.generated
    assert embeds[0, 1:5, 0].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert forwarded["position_ids"].shape == (3, 6)
    assert forwarded["max_length"] == 4
    assert "deepstack_embeds" not in forwarded


@pytest.mark.parametrize("clip_name,expected", [
    ("qwen35_2b", "qwen35"),
    ("qwen3vl_4b", "qwen3vl"),
    ("qwen3vl_8b", "qwen3vl"),
    ("gemma3_12b", "gemma"),
    ("llama", "llama3"),
])
def test_textgen_template_detection_uses_clip_tokenizer_identity(clip_name, expected):
    tokenizer = types.SimpleNamespace(clip_name=clip_name)
    clip = types.SimpleNamespace(tokenizer=tokenizer)
    assert textgen_nodes.detect_textgen_template(clip) == expected


def test_textgen_template_detection_uses_inner_tokenizer_class_fallback():
    Qwen3VLTokenizer = type("Qwen3VLTokenizer", (), {})
    tokenizer = types.SimpleNamespace(clip="encoder", encoder=Qwen3VLTokenizer())
    clip = types.SimpleNamespace(tokenizer=tokenizer)
    assert textgen_nodes.detect_textgen_template(clip) == "qwen3vl"


@pytest.mark.parametrize("clip", [types.SimpleNamespace(), types.SimpleNamespace(tokenizer=types.SimpleNamespace())])
def test_textgen_template_detection_rejects_unknown_clip(clip):
    with pytest.raises(ValueError, match="tokenizer"):
        textgen_nodes.detect_textgen_template(clip)


def test_qwen_template_families_use_exact_thinking_suppression():
    assert textgen_nodes.MODEL_TEMPLATES["qwen35"]["suppress_thinking"] == "<think>\n</think>\n"
    assert textgen_nodes.MODEL_TEMPLATES["qwen3vl"]["suppress_thinking"] is None
    assert textgen_nodes.MODEL_TEMPLATES["qwen35"]["visual_token"] == textgen_nodes.MODEL_TEMPLATES["qwen3vl"]["visual_token"]


def test_project_source_has_no_builtin_eval_calls():
    offenders = []
    tracked = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=CUSTOM_NODE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_paths = (
        CUSTOM_NODE_ROOT / relative_path
        for relative_path in tracked
        if pathlib.PurePosixPath(relative_path).parent == pathlib.PurePosixPath(".")
    )
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
                offenders.append(f"{source_path.name}:{node.lineno}")
    assert offenders == []
