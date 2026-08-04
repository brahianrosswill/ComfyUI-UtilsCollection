import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_text_nodes_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_text_nodes_test import text_nodes


def test_text_concatenate_autogrow_schema_uses_wildcard_links():
    schema = text_nodes.UC_TextConcatenateAutogrow.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    text_inputs = inputs["text_inputs"]

    assert schema.node_id == "UC_TextConcatenateAutogrow"
    assert schema.display_name == "Concatenate Text (Autogrow)"
    assert schema.category == "advanced/text"
    assert inputs["delimiter"].get_io_type() == "*"
    assert text_inputs.template.input.get_io_type() == "*"
    assert text_inputs.template.names == [
        f"text_{index}" for index in range(1, 101)
    ]
    assert text_inputs.template.min == 2
    assert schema.outputs[0].get_io_type() == "STRING"
    assert schema.outputs[0].display_name == "concatenated_text"


def test_text_concatenate_autogrow_joins_in_numeric_socket_order():
    output = text_nodes.UC_TextConcatenateAutogrow.execute(
        delimiter=" | ",
        text_inputs={
            "text_10": "ten",
            "text_2": "two",
            "text_1": "one",
        },
    )

    assert output.args == ("one | two | ten",)


def test_text_concatenate_autogrow_converts_arbitrary_values_to_strings():
    output = text_nodes.UC_TextConcatenateAutogrow.execute(
        delimiter=0,
        text_inputs={
            "text_1": 12,
            "text_2": True,
            "text_3": None,
        },
    )

    assert output.args == ("120True0None",)


def test_text_concatenate_autogrow_supports_empty_and_newline_delimiters():
    compact = text_nodes.UC_TextConcatenateAutogrow.execute(
        delimiter="",
        text_inputs={"text_1": "alpha", "text_2": "beta"},
    )
    multiline = text_nodes.UC_TextConcatenateAutogrow.execute(
        delimiter="\n",
        text_inputs={"text_1": "alpha", "text_2": "beta"},
    )

    assert compact.args == ("alphabeta",)
    assert multiline.args == ("alpha\nbeta",)
