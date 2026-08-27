import argparse
import json
from pathlib import Path
import runpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRESETS_PATH = REPOSITORY_ROOT / "vlm_presets.py"
VARS_PATH = REPOSITORY_ROOT / "vlm_presets_vars.py"
PREFIX_CARRIER = "\n\nBEGIN VIDEO REQUEST:\n"
SUFFIX_CARRIER = "\nEND VIDEO REQUEST.\n\n"
PRESETS = (
    (
        "h3_ref2va",
        "H3_REF2VA_PREFIX",
        "H3_REF2VA_SUFFIX",
    ),
    (
        "h3_ref2va_alt",
        "H3_REF2VA_ALT_PREFIX",
        "H3_REF2VA_ALT_SUFFIX",
    ),
    (
        "h3_mixed_ref2va",
        "H3_MIXED_REF2VA_PREFIX",
        "H3_MIXED_REF2VA_SUFFIX",
    ),
)


def serialized_pair(old: str, new: str, source: str) -> tuple[str, str]:
    candidates = (
        (json.dumps(old, ensure_ascii=False), json.dumps(new, ensure_ascii=False)),
        (json.dumps(old, ensure_ascii=True), json.dumps(new, ensure_ascii=True)),
        (repr(old), repr(new)),
    )
    for old_literal, new_literal in candidates:
        if source.count(old_literal) == 1:
            return old_literal, new_literal
    raise RuntimeError("Could not locate one exact serialized preset literal.")


def normalize(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def crlf(value: str) -> str:
    return normalize(value).replace("\n", "\r\n")


def readable_values(readable: dict, name: str, prefix_var: str, suffix_var: str):
    prefix = normalize(readable[prefix_var])
    suffix = normalize(readable[suffix_var])
    if not prefix.endswith(PREFIX_CARRIER):
        raise RuntimeError(f"{prefix_var} does not contain the request carrier.")
    if not suffix.startswith(SUFFIX_CARRIER):
        raise RuntimeError(f"{suffix_var} does not contain the request carrier.")
    raw = (
        prefix.removesuffix(PREFIX_CARRIER)
        + "\n\n"
        + suffix.removeprefix(SUFFIX_CARRIER)
    )
    return tuple(
        (dictionary, key, crlf(value))
        for dictionary, key, value in (
            ("system_query_additional_vlm", f"{name}_prefix", prefix),
            ("system_query_additional_vlm", f"{name}_suffix", suffix),
            ("system_query_raw_vlm", name, raw),
        )
    )


def insert_literal(source: str, dictionary: str, key: str, value: str) -> str:
    newline = "\r\n" if "\r\n" in source else "\n"
    if dictionary == "system_query_additional_vlm":
        marker = f"{newline}}}{newline}{newline}{newline}system_query_raw_vlm = {{"
    elif dictionary == "system_query_raw_vlm":
        marker = f"{newline}}}{newline}{newline}additional_instructions_vlm = {{"
    else:
        raise RuntimeError(f"Unsupported dictionary: {dictionary}")
    if source.count(marker) != 1:
        raise RuntimeError(f"Could not locate unique end of {dictionary}.")
    entry = (
        f"{newline}    {json.dumps(key)}: "
        f"{json.dumps(value, ensure_ascii=False)},"
    )
    return source.replace(marker, entry + marker, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    runtime = runpy.run_path(str(PRESETS_PATH))
    readable = runpy.run_path(str(VARS_PATH))
    source = PRESETS_PATH.read_bytes().decode("utf-8")
    updated_source = source
    changes: list[str] = []

    for name, prefix_var, suffix_var in PRESETS:
        for dictionary, key, expected in readable_values(
            readable, name, prefix_var, suffix_var
        ):
            current_original = runtime[dictionary].get(key)
            if current_original is None:
                updated_source = insert_literal(
                    updated_source, dictionary, key, expected
                )
                changes.append(f"add {dictionary}.{key}")
                continue
            if normalize(current_original) == normalize(expected):
                continue
            old_literal, new_literal = serialized_pair(
                current_original, expected, updated_source
            )
            updated_source = updated_source.replace(old_literal, new_literal, 1)
            changes.append(f"update {dictionary}.{key}")

    if not changes:
        print("Runtime H3 reference query presets are already synchronized.")
        return

    if args.apply:
        PRESETS_PATH.write_bytes(updated_source.encode("utf-8"))
        print("Synchronized H3 reference query presets.")
    else:
        print("Validated H3 reference query preset synchronization.")
    for change in changes:
        print(change)


if __name__ == "__main__":
    main()
