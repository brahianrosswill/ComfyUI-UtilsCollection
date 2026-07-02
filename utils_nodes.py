import sys
import random
from typing import Union
import json
import os
import torch
from comfy_api.latest import io
from comfy_extras.nodes_logic import SwitchNode, SoftSwitchNode

class UC_SwitchInverseNode(SwitchNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="UC_SwitchInverseNode",
            display_name="Switch (Inverse)",
            category="logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_true", template=template, lazy=True),
                io.MatchType.Input("on_false", template=template, lazy=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )


class UC_SoftSwitchInverseNode(SoftSwitchNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="UC_SoftSwitchInverseNode",
            display_name="Soft Switch (Inverse)",
            category="logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_true", template=template, lazy=True, optional=True),
                io.MatchType.Input("on_false", template=template, lazy=True, optional=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )

class UC_IntegerRangeRandom(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_IntegerRangeRandom",
            display_name="Random Integer in Range",
            category="utils/primitive",
            inputs=[
                io.Int.Input("minimum", min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("maximum", min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("seed", min=-sys.maxsize, max=sys.maxsize, control_after_generate=True),
            ],
            outputs=[io.Int.Output(display_name="random_integer")],
        )

    @classmethod
    def execute(cls, minimum: int, maximum: int, seed: int = 0) -> io.NodeOutput:
        min_val = min(minimum, maximum)
        max_val = max(minimum, maximum)
        rng = random.Random(seed)
        return io.NodeOutput(rng.randint(min_val, max_val))


class UC_TagNormalizeCombine(io.ComfyNode):
    """
    Node that normalizes scores in two sets of tags and combines them,
    deduplicating and sorting by the normalized scores.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_TagNormalizeCombine",
            display_name="Tag Normalize and Combine",
            category="advanced/text",
            inputs=[
                io.String.Input("tags_1", multiline=True, default=""),
                io.String.Input("tags_2", multiline=True, default=""),
                io.AnyType.Input(
                    "scores_1",
                    tooltip="Dictionary of scores for tags_1",
                    optional=True,
                ),
                io.AnyType.Input(
                    "scores_2",
                    tooltip="Dictionary of scores for tags_2",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="deduped_tags"),
                io.AnyType.Output(display_name="normalized_scores"),
            ],
        )

    @staticmethod
    def normalize_scores(scores: dict, min_val=0.000001, max_val=0.999999) -> dict:
        if not scores:
            return {}

        current_scores = [float(v) for v in scores.values()]
        current_min = min(current_scores)
        current_max = max(current_scores)

        if current_max == current_min:
            return {k: max_val for k in scores.keys()}

        normalized = {}
        for k, v in scores.items():
            norm = min_val + (float(v) - current_min) / (current_max - current_min) * (
                max_val - min_val
            )
            normalized[k] = norm
        return normalized

    @classmethod
    def execute(
        cls, tags_1: Union[str, list], tags_2: Union[str, list], scores_1: Union[str, dict] = None, scores_2: Union[str, dict] = None
    ) -> io.NodeOutput:
        # Parse tags
        def parse_tags(t_input):
            if isinstance(t_input, list):
                return [str(t).strip() for t in t_input if t]
            if not t_input or not isinstance(t_input, str):
                return []
            # Split by comma and handle potential spaces
            return [t.strip() for t in t_input.split(",") if t.strip()]

        t1_list = parse_tags(tags_1)
        t2_list = parse_tags(tags_2)

        # Handle scores
        def process_scores(s_input, t_list):
            if s_input is None:
                # Generate even distribution
                if not t_list:
                    return {}
                num_tags = len(t_list)
                if num_tags == 1:
                    return {t_list[0]: 0.999999}

                max_v = 0.999999
                min_v = 0.000001
                scores = {}
                for i, tag in enumerate(t_list):
                    # Linearly interpolate from max_v down to min_v
                    score = max_v - i * (max_v - min_v) / (num_tags - 1)
                    scores[tag] = score
                return scores

            # Parse existing scores
            def parse_s(s_in):
                if isinstance(s_in, dict):
                    return s_in
                if not s_in or not isinstance(s_in, str):
                    return {}
                try:
                    return json.loads(s_in)
                except json.JSONDecodeError:
                    return {}

            return cls.normalize_scores(parse_s(s_input))

        norm_s1 = process_scores(scores_1, t1_list)
        norm_s2 = process_scores(scores_2, t2_list)

        # Combine and deduplicate
        combined_scores = {}

        # Process first set
        for t in t1_list:
            score = norm_s1.get(t, 0.000001)
            combined_scores[t] = score

        # Process second set with deduplication logic
        for t in t2_list:
            score = norm_s2.get(t, 0.000001)
            if t in combined_scores:
                # Keep the one with the highest normalized score
                if score > combined_scores[t]:
                    combined_scores[t] = score
            else:
                combined_scores[t] = score

        # Sort tags by normalized scores (descending)
        sorted_tags = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Prepare outputs
        deduped_tags_str = ", ".join(sorted_tags)
        normalized_scores_dict = {tag: combined_scores[tag] for tag in sorted_tags}

        return io.NodeOutput(deduped_tags_str, normalized_scores_dict)


class UC_RandInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_RandInt",
            display_name="RandomInt",
            category="utils/primitive",
            inputs=[
                io.Int.Input("value", min=-sys.maxsize, max=sys.maxsize, control_after_generate=True),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)


class UC_StaticInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_StaticInt",
            display_name="StaticInt",
            category="utils/primitive",
            inputs=[
                io.Int.Input("value", min=-sys.maxsize, max=sys.maxsize),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)


class UC_RandIntRange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_RandIntRange",
            display_name="RandomIntRange",
            category="utils/primitive",
            inputs=[
                io.Int.Input("min", default=0, min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("max", default=100, min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, min: int, max: int, seed: int) -> io.NodeOutput:
        rng = random.Random(seed)
        return io.NodeOutput(rng.randint(min, max))


class UC_ColorConvertNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ColorConvertNode",
            display_name="Color Convert (Bottom of node is color selector)",
            category="advanced/color",
            inputs=[
                io.Combo.Input("from_mode", options=["auto", "Manual Hex #FFFFFF", "Int 0-16777215", "Comma separated 255,255,255"], default="auto", tooltip="Select how to interpret the color input. 'Auto' will use the color picker input unless one of the other fields is filled in, in which case it will use the filled-in field based on the other options. The other modes will take precedence over the color picker input when their respective fields are filled in."),
                io.Color.Input("color_hex", default="#FFFFFF", display_name="This is not the Color Selector →", tooltip="Select a color using the color picker. This input is used when 'from_mode' is set to 'auto' and no other manual inputs are provided. The selected color will be converted to hex, int, and string formats for output."),
                io.String.Input("manual_hex", multiline=False, optional=True, display_name="Manual Hex Input", tooltip="Enter a hex color code manually, e.g. #FF00FF. Takes precedence over the color picker input if 'from_mode' is set to 'Manual Hex #FFFFFF'."),
                io.Int.Input("color_int", min=-1, max=16777215, default=-1, optional=True, display_name="Color Int Input", tooltip="Enter a color as an integer (0-16777215). Interpreted as 0xRRGGBB. Takes precedence over the color picker input if 'from_mode' is set to 'Int 0-16777215'."),
                io.String.Input("comma_separated", multiline=False, optional=True, display_name="Comma Separated Input", tooltip="Enter a color as comma-separated RGB values (0-255), e.g. 255,0,255. Takes precedence over the color picker input if 'from_mode' is set to 'Comma separated 255,255,255'."),
            ],
            outputs=[
                io.String.Output(display_name="color_hex_output"),
                io.Int.Output(display_name="color_int_output"),
                io.String.Output(display_name="color_string_output"),
            ]
        )

    @classmethod
    def _validate_hex(cls, s):
        """Return (r, g, b) if s is a valid 6-digit hex string (case-insensitive, # prefix optional), else None."""
        if s is None:
            return None
        s = s.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            return None
        if not all(c in "0123456789abcdefABCDEF" for c in s):
            return None
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)

    @classmethod
    def _validate_int(cls, n):
        """Return (r, g, b) if n is an int in 0–16777215, else None."""
        if n is None or not isinstance(n, int):
            return None
        if not (0 <= n <= 16777215):
            return None
        return ((n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF)

    @classmethod
    def _validate_csv(cls, s):
        """Return (r, g, b) if s is 3 ints 0–255 separated by ',' or ', ', else None."""
        if s is None:
            return None
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            return None
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            return None
        if not all(0 <= v <= 255 for v in vals):
            return None
        return tuple(vals)

    @classmethod
    def _rgb_to_outputs(cls, r, g, b):
        return (
            f"#{r:02X}{g:02X}{b:02X}",
            (r << 16) | (g << 8) | b,
            f"{r}, {g}, {b}",
        )

    @classmethod
    def execute(cls, from_mode, color_hex, manual_hex=None, color_int=None, comma_separated=None) -> io.NodeOutput:
        # Sanitize sentinels to None
        if color_int is None or not (0 <= color_int <= 16777215):
            color_int = None
        if not manual_hex or manual_hex.strip() == "":
            manual_hex = None
        if not comma_separated or comma_separated.strip() == "":
            comma_separated = None

        if from_mode == "auto":
            valid_hex = cls._validate_hex(manual_hex)
            valid_int = cls._validate_int(color_int)
            valid_csv = cls._validate_csv(comma_separated)
            valid_inputs = [x for x in [valid_hex, valid_int, valid_csv] if x is not None]

            if len(valid_inputs) == 1:
                r, g, b = valid_inputs[0]
            else:
                if len(valid_inputs) > 1:
                    print("[ColorConvertNode] Warning: multiple optional inputs are valid in auto mode, falling back to color picker.")
                # Use picker (len==0 or len>1)
                picker = cls._validate_hex(color_hex)
                if picker is None:
                    raise ValueError(f"Color picker value '{color_hex}' must be in format #RRGGBB with valid hex digits")
                r, g, b = picker

            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Manual Hex #FFFFFF":
            result = cls._validate_hex(manual_hex)
            if result is None:
                raise ValueError(f"Manual hex input '{manual_hex}' must be 6 valid hex digits (0-9, a-f, A-F), with optional # prefix")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Int 0-16777215":
            result = cls._validate_int(color_int)
            if result is None:
                raise ValueError(f"Color integer input '{color_int}' must be in range 0–16777215")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Comma separated 255,255,255":
            result = cls._validate_csv(comma_separated)
            if result is None:
                raise ValueError(f"Comma-separated input '{comma_separated}' must be 3 integers 0–255 separated by ',' or ', '")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        else:
            raise ValueError(f"Unknown from_mode '{from_mode}'")

        return io.NodeOutput(color_hex_output, color_int_output, color_string_output)


class UC_ExtractBoundingBox(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ExtractBoundingBox",
            display_name="Extract Bounding Box",
            category="utils/primitive",
            inputs=[
                io.AnyType.Input(
                    "input_data",
                    tooltip="Input data containing bounding boxes (JSON string, list, dict, or nested structure)"
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Index of the bounding box to extract"
                ),
            ],
            outputs=[
                io.Int.Output(display_name="x"),
                io.Int.Output(display_name="y"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def find_boxes(cls, data) -> list:
        boxes = []
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass

        if isinstance(data, dict):
            if all(k in data for k in ("x", "y", "width", "height")):
                boxes.append(data)
            else:
                for v in data.values():
                    boxes.extend(cls.find_boxes(v))
        elif isinstance(data, (list, tuple)):
            for item in data:
                boxes.extend(cls.find_boxes(item))

        return boxes

    @classmethod
    def execute(cls, input_data: any, index: int) -> io.NodeOutput:
        boxes = cls.find_boxes(input_data)
        if not boxes:
            raise ValueError("No bounding boxes containing 'x', 'y', 'width', and 'height' were found in the input data.")

        if index < 0 or index >= len(boxes):
            raise ValueError(f"Index {index} is out of range. Found {len(boxes)} bounding box(es).")

        box = boxes[index]

        try:
            x = int(float(box["x"]))
            y = int(float(box["y"]))
            w = int(float(box["width"]))
            h = int(float(box["height"]))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert bounding box values at index {index} to integers: {e}")

        return io.NodeOutput(x, y, w, h)


class UC_Krea2LayerProbe(io.ComfyNode):
    """
    Krea 2 Text Encoder Activation Probing Node.

    This node unpacks the flattened 12-layer text conditioning tensor of shape (B, seq, 30720)
    back into its original 12 tapped components of shape (B, 12, seq, 2560) representing the
    last 12 layers of Qwen3-VL-4B.

    It calculates layer-wise activation statistics (Mean, Standard Deviation, Max, Min, L2 Norm)
    and saves them in a JSONL file to compare safe and refused prompt dynamics. It can also save
    sequence-averaged raw activation tensors to build offline datasets for pinpoint weight-level
    ablation.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Krea2LayerProbe",
            display_name="Krea 2 Layer Probe",
            category="advanced/conditioning",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input(
                    "prompt_label",
                    default="safe_prompt",
                    tooltip="A unique name/tag to log this prompt in the stats. Tag safe prompts with 'safe_...' and refused prompts with 'refused_...' to perform differential analysis."
                ),
                io.String.Input(
                    "log_dir",
                    default="krea2_stats",
                    tooltip="The directory where JSONL statistical logs and optional raw tensor files (.pt) will be written."
                ),
                io.Boolean.Input(
                    "save_activations",
                    default=False,
                    tooltip="If True, saves raw 12-layer sequence-averaged activation tensors as .pt files in the log directory for difference-vector computations."
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="statistics_summary"),
            ],
        )

    @classmethod
    def execute(cls, conditioning, prompt_label: str = "safe_prompt", log_dir: str = "krea2_stats", save_activations: bool = False) -> io.NodeOutput:
        # Conditioning is a list of tuples: [(tensor, {"pooled_output": pooled})]
        cond_tensor = conditioning[0][0]  # Shape: (B, seq, 30720)

        B, seq, total_dim = cond_tensor.shape
        num_layers = 12
        hidden_size = 2560

        if total_dim != num_layers * hidden_size:
            raise ValueError(f"Expected conditioning dimension {num_layers * hidden_size}, got {total_dim}")

        # Unpack layers: (B, seq, 12, 2560) -> (B, 12, seq, 2560)
        unpacked = cond_tensor.view(B, seq, num_layers, hidden_size).permute(0, 2, 1, 3)

        # Calculate Layer-wise Stats
        stats = {}
        os.makedirs(log_dir, exist_ok=True)

        for i in range(num_layers):
            layer_act = unpacked[:, i, :, :]  # (B, seq, 2560)

            mean_val = torch.mean(layer_act).item()
            std_val = torch.std(layer_act).item()
            max_val = torch.max(layer_act).item()
            min_val = torch.min(layer_act).item()
            l2_norm = torch.norm(layer_act, p=2, dim=-1).mean().item()

            stats[f"layer_{i}"] = {
                "mean": mean_val,
                "std": std_val,
                "max": max_val,
                "min": min_val,
                "l2_norm": l2_norm
            }

        # Write summary to log
        log_file = os.path.join(log_dir, "probe_log.jsonl")
        log_entry = {
            "prompt_label": prompt_label,
            "seq_len": seq,
            "stats": stats
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Optionally save raw tensor activations to calculate average directions later
        if save_activations:
            avg_activation = torch.mean(unpacked, dim=2)  # Shape: (B, 12, 2560)
            save_path = os.path.join(log_dir, f"act_{prompt_label}.pt")
            torch.save(avg_activation.cpu(), save_path)

        summary_str = f"Probed {seq} tokens across {num_layers} layers.\n"
        summary_str += f"Layer 7 (Max weight projection) L2: {stats['layer_7']['l2_norm']:.4f}, Max: {stats['layer_7']['max']:.4f}"

        return io.NodeOutput(conditioning, summary_str)


class UC_Krea2LayerAblator(io.ComfyNode):
    """
    Krea 2 Text Encoder Activation Pinpoint Ablator.

    This node loads pre-computed difference vectors representing the shift in activation spaces
    during safety refusals, and performs a clean orthogonal projection to subtract the refusal
    direction component.

    Warning: As direct activation manipulation (swapping/clamping/orthogonal subtraction) can
    have subtle side-effects on photographic style, this node is primarily an analytical testbed.
    Using the analytical probing results to surgically ablate weights on the diff LoRA is
    the recommended production approach.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Krea2LayerAblator",
            display_name="Krea 2 Layer Pinpoint Ablator",
            category="advanced/conditioning",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input(
                    "vectors_path",
                    default="krea2_stats/refusal_directions.pt",
                    tooltip="Path to the .pt file containing pre-computed difference vectors for each of the 12 tapped layers."
                ),
                io.Float.Input(
                    "ablation_strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Ablation scale. 1.0 performs pure orthogonal projection (subtraction of the refusal vector component)."
                ),
                io.String.Input(
                    "layers_mask",
                    default="0,0,0,0,0,0,0,1,1,1,1,0",
                    tooltip="12 comma-separated binary integers (0 or 1) selecting which layers undergo orthogonal projection (e.g., '0,0,0,0,0,0,0,1,1,1,1,0' to target deep layers)."
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning, vectors_path: str = "krea2_stats/refusal_directions.pt", ablation_strength: float = 1.0, layers_mask: str = "0,0,0,0,0,0,0,1,1,1,1,0") -> io.NodeOutput:
        if not os.path.exists(vectors_path):
            print(f"Warning: Refusal vectors file not found at {vectors_path}. Skipping ablation.")
            return io.NodeOutput(conditioning)

        refusal_vectors = torch.load(vectors_path).to(torch.float32)

        mask = [int(x.strip()) for x in layers_mask.split(",")]
        if len(mask) != 12:
            raise ValueError("Layers mask must contain exactly 12 comma-separated binary digits (0 or 1)")

        modified_cond = []
        for cond_tensor, extra in conditioning:
            B, seq, total_dim = cond_tensor.shape
            num_layers = 12
            hidden_size = 2560

            unpacked = cond_tensor.view(B, seq, num_layers, hidden_size).permute(0, 2, 1, 3).clone()

            device = cond_tensor.device
            dtype = cond_tensor.dtype

            for i in range(num_layers):
                if mask[i] == 1:
                    v_refuse = refusal_vectors[0, i, :].to(device=device, dtype=torch.float32)

                    norm = torch.norm(v_refuse, p=2)
                    if norm > 1e-6:
                        v_hat = v_refuse / norm

                        layer_act = unpacked[:, i, :, :].to(dtype=torch.float32)

                        dot_product = torch.sum(layer_act * v_hat, dim=-1, keepdim=True)

                        projection = dot_product * v_hat
                        ablated = layer_act - ablation_strength * projection

                        unpacked[:, i, :, :] = ablated.to(dtype=dtype)

            repacked = unpacked.permute(0, 2, 1, 3).reshape(B, seq, total_dim)

            new_extra = extra.copy()
            modified_cond.append((repacked, new_extra))

        return io.NodeOutput(modified_cond)


class UC_EncoderNodesGuide(io.ComfyNode):
    """
    Detailed Markdown formatted documentation and guide for advanced, plus, and scaled-bias encoder nodes.
    Choose topics to view in any Markdown rendering node.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_EncoderNodesGuide",
            display_name="Encoder Nodes Guide",
            category="utils/documentation",
            inputs=[
                io.Combo.Input(
                    "topic",
                    options=[
                        "system_prompt",
                        "image_input_how_it_works",
                        "scaled_bias_and_weighting",
                        "math_expressions",
                        "saving_embeddings",
                    ],
                    default="system_prompt",
                    tooltip="Select the topic you would like to view documentation for.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="markdown"),
            ],
        )

    @classmethod
    def execute(cls, topic: str) -> io.NodeOutput:
        if topic == "system_prompt":
            markdown = (
                "##### System Prompt Guide\n"
                "The System Prompt is a high-level instruction injected at the very beginning of the chat template "
                "(before user prompt tokenization). It guides the behavior, style, and tone of the underlying Vision-Language Model (VLM).\n\n"
                "##### Key Details:\n"
                "- If system_prompt is left empty, the node falls back to a default high-quality description prompt (e.g., the detailed Krea2 visual formatting template).\n"
                "- When a custom system_prompt is provided, it completely overrides the default block instructions (for example, instructing the VLM to focus exclusively on specific subjects, lighting styles, or color palettes).\n"
                "- Safe concatenation templates are utilized to format the system, user, and assistant turns cleanly, ensuring special characters such as brackets or parentheses are never corrupted."
            )
        elif topic == "image_input_how_it_works":
            markdown = (
                "##### How Image Inputs Work\n"
                "All advanced encoder nodes allow you to dynamically load multiple images through dynamic Autogrow inputs.\n\n"
                "##### Key Details:\n"
                "- When you connect images to the 'image_inputs' slot, they are automatically parsed.\n"
                "- For advanced math blending nodes, the connected active images are dynamically mapped to sequential, contiguous letter variables: `a`, `b`, `c`, `d`, etc., starting from `a` for the first active connected image.\n"
                "- Since the variables are assigned sequentially to only the active connections, empty/unconnected slots are skipped entirely, eliminating any missing variable errors.\n"
                "- The node performs individual, sequential encoding passes for each active image using the prompt, and maps the resulting conditioning tensors to their respective letter variable.\n"
                "- All images are resized to the set `vlm_resolution` (e.g. 'Fast (384)', 'Balanced (512)', etc.) using high-quality upscaling/downscaling."
            )
        elif topic == "scaled_bias_and_weighting":
            markdown = (
                "##### Scaled Bias & Weighting Syntax\n"
                "Standard ComfyUI weighting (using parentheses and colons) is not natively supported by the custom tokenizers of advanced models like Qwen or Gemma. To bridge this gap, our Scaled Bias nodes implement a classical weight translation engine.\n\n"
                "##### Key Details:\n"
                "- You can write your weights using standard weighting syntax: `(prompt text:weight)`, for example `(beautiful sunset:1.25)` or `(red car:0.8)`.\n"
                "- Before tokenization, our weight translation engine parses and extracts these markers, strips the outer parenthesis and weight markers, and compiles the clean text.\n"
                "- To handle the complex visual/image token expansions precisely, a dynamic token-to-embedding mapping calculates the exact sequence locations in the final encoded embedding tensor.\n"
                "- The precise slices of the conditioning tensor are then multiplied element-wise by the target strength.\n"
                "- The pooled output (the global embedding representing the entire sequence) is also safely scaled by the maximum weight found."
            )
        elif topic == "math_expressions":
            markdown = (
                "##### Math Expressions for Conditioning Blending\n"
                "Mathematical conditioning blending allows you to perform continuous element-wise PyTorch mathematical operations directly in the CLIP/VLM conditioning embedding space (latent vector space) inside a single node, exactly mimicking the behavior of CES Conditioning Formula nodes.\n\n"
                "##### Key Details:\n"
                "- Enter your mathematical expression directly in the dedicated single-line `formula` input field (e.g., `(a + b) / 2`). No prompt pipe syntax is required.\n"
                "- Inside the formula, use variables `a`, `b`, `c`, `d`... representing your active connected VLM image conditionings.\n"
                "- Under the hood, the node automatically runs independent, native single-image encoding passes for each active image, then evaluates the mathematical formula directly on those extracted high-dimensional continuous sequence tensors ($C \\in \\mathbb{R}^{B \\times L \\times D}$), pooled tensors ($P \\in \\mathbb{R}^{B \\times D}$), and DeepStack layers.\n"
                "- Supported operations:\n"
                "  - Addition (+), Subtraction (-), Multiplication (*), Division (/)\n"
                "  - Parentheses for nesting operations: `((a * 1.05) + b) / 2`\n"
                "  - Functions: `clamp(tensor, min, max)`, `min(tensor1, tensor2)`, `max(tensor1, tensor2)`, `abs(tensor)`\n"
                "- **Nested Weights Support**: Inside the math expression, you can use classical weight syntax like `(a:1.2) + (b:0.8)`. Under the hood, these are dynamically preprocessed into scalar embedding multiplications `(a * 1.2)` and `(b * 0.8)` respectively before evaluation.\n"
                "- **Sequence Alignment Modes**: If your images have different aspect ratios or resolutions, the node supports two alignment options below the multiplier widget:\n"
                "  - `zero-pad`: Silently zero-pads shorter sequence tensors to align lengths (matches ComfyUI core conditioning logic exactly).\n"
                "  - `interpolate`: Dynamically resizes the visual token sequence using 1D linear interpolation to align attention features perfectly across the entire sequence without dead space.\n"
                "- Security: All formulas are parsed and evaluated within a completely sandboxed namespace (`__builtins__ = {}`), preventing any insecure code executions while giving you full access to PyTorch's tensor math."
            )
        elif topic == "saving_embeddings":
            markdown = (
                "##### Saving Pre-Transformer Input Embeddings\n"
                "You can save raw pre-transformer interleaved text and visual embeddings directly to disk before they enter the transformer layers.\n\n"
                "##### Key Details:\n"
                "- The **Krea 2 Input Embeddings** (`UC_Krea2InputEmbeds`) and **Qwen3-VL Unified Input Embeddings** (`UC_Qwen3VLInputEmbeds`) nodes handle this task.\n"
                "- The prompt text is tokenized with `skip_template=True` so that any model-specific chat/prompt wrappers are skipped, preserving raw text embeddings.\n"
                "- If an image is connected, the image is tokenized and its visual token pad structures are interleaved within the language tokens.\n"
                "- The model's `process_tokens` method extracts the high-dimensional continuous input embeddings.\n"
                "- Slicing can be toggled optionally via the `slice_visual_tokens` widget (Method A) to extract pure language/text embedding tensors. If disabled (default), the raw interleaved sequence is preserved.\n"
                "- Tensors are written as a `.safetensors` file under your specified name in ComfyUI's `embeddings/` folder."
            )
        else:
            markdown = "Unknown topic selected."

        return io.NodeOutput(markdown)


EncoderNodesGuide = UC_EncoderNodesGuide

