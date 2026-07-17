import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_primitive_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_primitive_test.utils_nodes import UC_StaticFloat, UC_StaticInt
finally:
    cli_args.cpu = prior_cpu


def test_static_float_schema_supports_precise_shared_values():
    schema = UC_StaticFloat.define_schema()
    value = schema.inputs[0]

    assert schema.node_id == "UC_StaticFloat"
    assert schema.category == "utils/primitive"
    assert value.default == 1.0
    assert value.step == 0.01
    assert value.min == -sys.float_info.max
    assert value.max == sys.float_info.max
    assert UC_StaticFloat.execute(1.2345).result == (1.2345,)


def test_static_integer_remains_the_numeric_pair_compatibility_node():
    schema = UC_StaticInt.define_schema()

    assert schema.node_id == "UC_StaticInt"
    assert schema.category == "utils/primitive"
    assert UC_StaticInt.execute(7).result == (7,)
