from comfy_api.latest import io

from .qwen_vlm_presets import qwen_system_instructions_vlm


class UC_QwenVLMSysInstrPresets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = sorted(qwen_system_instructions_vlm)
        return io.Schema(
            node_id="UC_QwenVLMSysInstrPresets",
            display_name="Qwen VLM System Instruction Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    display_name="qwen_vlm_system_instruction_preset",
                    options=options,
                    default=options[0] if options else "",
                ),
            ],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        return io.NodeOutput(qwen_system_instructions_vlm.get(preset, ""))


class UC_QwenVLMSysInstrAdvPresets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = sorted(qwen_system_instructions_vlm)
        return io.Schema(
            node_id="UC_QwenVLMSysInstrAdvPresets",
            display_name="Qwen VLM System Instruction Advanced Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    display_name="qwen_vlm_system_instruction_advanced_preset",
                    options=options,
                    default=options[0] if options else "",
                ),
                io.String.Input("system_query", multiline=True, default=""),
                io.String.Input("user_query", multiline=True, default=""),
            ],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset, system_query, user_query) -> io.NodeOutput:
        result = qwen_system_instructions_vlm.get(preset, "")
        if user_query and user_query.strip():
            result += "\n\nRequested transformation and output requirements:\n" + user_query.strip()
        if system_query and system_query.strip():
            result += "\n\nHighest-priority system override:\n" + system_query.strip()
        return io.NodeOutput(result)
