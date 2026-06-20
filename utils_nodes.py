import sys
import random
from typing import Union
import json
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

