from comfy_api.latest import io
from .helper_functions import (
    join_words_in_text,
    to_bold_fraktur_style,
    from_bold_fraktur_style,
    remove_joiners,
)

class UC_BoldFrakturTextStyle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_BoldFrakturTextStyle",
            display_name="Bold Fraktur Text style",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter text to style...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="fraktur_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = to_bold_fraktur_style(text)
        return io.NodeOutput(result)


class UC_UnBoldFrakturTextStyle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_UnBoldFrakturTextStyle",
            display_name="UnBoldFrakturTextStyle",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter styled text to convert back...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="plain_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = from_bold_fraktur_style(text)
        return io.NodeOutput(result)


class UC_WordJoiner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_WordJoiner",
            display_name="Word Joiner",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter text to join...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="joined_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = join_words_in_text(text)
        return io.NodeOutput(result)


class UC_UnWordJoiner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_UnWordJoiner",
            display_name="Remove Word Joiners",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter text with joiners...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="unjoined_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = remove_joiners(text)
        return io.NodeOutput(result)
