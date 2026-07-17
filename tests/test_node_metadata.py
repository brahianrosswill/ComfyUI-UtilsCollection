import asyncio
import importlib.util
import pathlib
import sys


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_metadata_test"


def _load_extension_package():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        CUSTOM_NODE_ROOT / "__init__.py",
        submodule_search_locations=[str(CUSTOM_NODE_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(PACKAGE_NAME, package)
    spec.loader.exec_module(package)
    return package


def test_registered_non_deprecated_nodes_have_search_metadata():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    checked = []

    for node_class in node_classes:
        schema = node_class.define_schema()
        if schema.is_deprecated or "(Legacy)" in (schema.display_name or ""):
            continue
        checked.append(schema.node_id)
        assert schema.description and not schema.description.startswith("Provides UC_")
        assert schema.search_aliases
        assert len(schema.search_aliases) == len(set(schema.search_aliases))

    assert "UC_StaticFloat" in checked
    assert "UC_ConditioningConsensusBlend" in checked


def test_legacy_nodes_do_not_inherit_canonical_search_metadata():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    legacy_static = next(node for node in node_classes if node.__name__ == "StaticInt")
    schema = legacy_static.define_schema()

    assert "(Legacy)" in schema.display_name
    assert not schema.search_aliases
