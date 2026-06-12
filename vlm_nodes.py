import os
import json

from comfy_api.latest import ComfyExtension, io
from .presets_collection import system_instructions_vlm, system_query_additional_vlm, additional_instructions_vlm

class UC_VLMSysInstrPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        json_path = os.path.join(os.path.dirname(__file__), "system_instructions_vlm.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                presets = json.load(f)
        except Exception as e:
            print(f"Error loading VLMSysInstrPresets: {e}")
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(list(presets.keys()))
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrPresets",
            display_name="VLM System Instruction Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=options,
                    default=default,
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_instruction"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        system_instruction = presets_dict.get(preset, "")
        return io.NodeOutput(system_instruction)


class UC_VLMSysQueryAddPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        base_names = set()
        json_path = os.path.join(os.path.dirname(__file__), "system_query_additional_vlm.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key in data:
                    base_names.add(key.removesuffix("_prefix").removesuffix("_suffix"))
        except Exception as e:
            print(f"Error loading VLMSysQueryAddPresets: {e}")
        return sorted(list(base_names))

    @classmethod
    def define_schema(cls):
        options = cls.get_presets()
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysQueryAddPresets",
            display_name="VLM System Query Add Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=options,
                    default=default,
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_query_additional"),
            ],
        )

    @classmethod
    def execute(cls, preset, text) -> io.NodeOutput:
        json_path = os.path.join(os.path.dirname(__file__), "system_query_additional_vlm.json")
        prefix = ""
        suffix = ""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                prefix = data.get(f"{preset}_prefix", "")
                suffix = data.get(f"{preset}_suffix", "")
        except Exception as e:
            print(f"Error executing VLMSysQueryAddPresets: {e}")

        return io.NodeOutput(f"{prefix}{text}{suffix}")


class UC_VLMSysInstrAdvPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        json_path = os.path.join(
            os.path.dirname(__file__), "system_instructions_vlm.json"
        )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                presets = json.load(f)
        except Exception as e:
            print(f"Error loading VLMSysInstrPresets: {e}")
        return presets

    @classmethod
    def get_additional_instructions(cls):
        instructions = {}
        json_path = os.path.join(
            os.path.dirname(__file__), "additional_instructions_vlm.json"
        )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                instructions = json.load(f)
        except Exception as e:
            print(f"Error loading additional_instructions_vlm: {e}")
        return instructions

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(list(presets.keys()))
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrAdvPresets",
            display_name="VLM System Instruction Advanced Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=options,
                    default=default,
                ),
                io.Boolean.Input("jailbreak", default=False),
                io.String.Input("system_query", multiline=True, default=""),
                io.String.Input("user_query", multiline=True, default=""),
            ],
            outputs=[
                io.String.Output(display_name="system_instruction"),
            ],
        )

    @classmethod
    def execute(cls, preset, jailbreak, system_query, user_query) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        additional = cls.get_additional_instructions()

        system_instruction = presets_dict.get(preset, "")

        # Start building the final output
        # Formula: {jailbreak_prefix} + {chosen_preset_content} + {system_query_prefix} + {system_query_string} + {user_query_prefix} + {user_query_string} + {user_query_suffix} + {system_query_suffix} + {jailbreak_suffix}

        result = system_instruction

        # Apply system query wrapping
        if system_query:
            sq_prefix = additional.get("system_query_prefix", "")
            sq_suffix = additional.get("system_query_suffix", "")
            result = result + sq_prefix + system_query

            # Apply user query nesting
            if user_query:
                uq_prefix = additional.get("user_query_prefix", "")
                uq_suffix = additional.get("user_query_suffix", "")
                result = result + uq_prefix + user_query + uq_suffix

            result = result + sq_suffix
        elif user_query:
            # If system_query is empty but user_query is not, just add user query components
            uq_prefix = additional.get("user_query_prefix", "")
            uq_suffix = additional.get("user_query_suffix", "")
            result = result + uq_prefix + user_query + uq_suffix

        # Apply jailbreak wrapping
        if jailbreak:
            jb_prefix = additional.get("jailbreak_prefix", "")
            jb_suffix = additional.get("jailbreak_suffix", "")
            result = jb_prefix + result + jb_suffix

        return io.NodeOutput(result)
