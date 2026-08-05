from comfy_api.latest import io

from .minimax_h3_vlm_presets import (
    minimax_h3_system_instructions_vlm,
    minimax_h3_vlm_jailbreak_prefix,
    minimax_h3_vlm_jailbreak_suffix,
)


class UC_MiniMaxH3VLMSysInstrPresets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = list(minimax_h3_system_instructions_vlm)
        return io.Schema(
            node_id="UC_MiniMaxH3VLMSysInstrPresets",
            display_name="MiniMax H3 VLM System Instruction Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    display_name="minimax_h3_vlm_system_instruction_preset",
                    options=options,
                    default=options[0] if options else "",
                ),
            ],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        return io.NodeOutput(minimax_h3_system_instructions_vlm.get(preset, ""))


class UC_MiniMaxH3VLMSysInstrAdvPresets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = list(minimax_h3_system_instructions_vlm)
        return io.Schema(
            node_id="UC_MiniMaxH3VLMSysInstrAdvPresets",
            display_name="MiniMax H3 VLM System Instruction Advanced Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    display_name="minimax_h3_vlm_system_instruction_advanced_preset",
                    options=options,
                    default=options[0] if options else "",
                ),
                io.String.Input("system_query", multiline=True, default=""),
                io.String.Input("user_query", multiline=True, default=""),
                io.Boolean.Input("jailbreak", default=False),
            ],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset, system_query, user_query, jailbreak=False) -> io.NodeOutput:
        result = minimax_h3_system_instructions_vlm.get(preset, "")
        if jailbreak:
            result = minimax_h3_vlm_jailbreak_prefix + "\n\n" + result
        if user_query and user_query.strip():
            result += "\n\nRequested target-video requirements:\n" + user_query.strip()
        if jailbreak:
            result += "\n\n" + minimax_h3_vlm_jailbreak_suffix
        if system_query and system_query.strip():
            result += "\n\nHighest-priority system override:\n" + system_query.strip()
        return io.NodeOutput(result)
