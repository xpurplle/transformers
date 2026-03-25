# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TRF014: Models with non-empty _tied_weights_keys must have tie_word_embeddings in their Config."""

import ast
from pathlib import Path

from ._helpers import Violation, _get_class_assignments, _simple_name, full_name, iter_pretrained_classes


RULE_ID = ""  # Set by discovery

_PRETRAINED_CONFIG_NAMES = {"PreTrainedConfig", "PretrainedConfig"}


def _is_non_empty_collection(node: ast.AST) -> bool:
    """Return True if the AST node is a non-empty Dict, List, Set, or Tuple literal."""
    if isinstance(node, ast.Dict):
        return len(node.keys) > 0
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return len(node.elts) > 0
    return False


def _config_has_tie_word_embeddings(config_path: Path) -> bool:
    """Parse a configuration file and check if any config class defines or inherits tie_word_embeddings."""
    try:
        source = config_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(config_path))
    except (OSError, SyntaxError):
        return True  # Don't flag if we can't read/parse the config

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        # If the config inherits from a non-PreTrainedConfig base (e.g. MistralConfig),
        # it likely inherits tie_word_embeddings from the parent model config.
        for base in node.bases:
            try:
                base_name = _simple_name(full_name(base))
            except ValueError:
                continue
            if base_name not in _PRETRAINED_CONFIG_NAMES and base_name.endswith("Config"):
                return True

        # If the config uses sub_configs with a text_config, the text sub-config
        # handles tie_word_embeddings (e.g. composite vision-language models).
        assignments = _get_class_assignments(node)
        sub_configs = assignments.get("sub_configs")
        if isinstance(sub_configs, ast.Dict):
            for key in sub_configs.keys:
                if isinstance(key, ast.Constant) and key.value == "text_config":
                    return True

        # Check class-level assignments (both plain and annotated)
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id == "tie_word_embeddings":
                    return True
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "tie_word_embeddings":
                        return True
            # Check self.tie_word_embeddings = ... inside methods
            if isinstance(item, ast.FunctionDef):
                for stmt in ast.walk(item):
                    if (
                        isinstance(stmt, ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Attribute)
                        and isinstance(stmt.targets[0].value, ast.Name)
                        and stmt.targets[0].value.id == "self"
                        and stmt.targets[0].attr == "tie_word_embeddings"
                    ):
                        return True
    return False


def _find_config_file(file_path: Path) -> Path | None:
    """Given a modeling/modular file, find the corresponding configuration file.

    Tries to match the suffix first (modeling_foo_bar.py -> configuration_foo_bar.py),
    then falls back to any configuration file in the same directory.
    """
    model_dir = file_path.parent
    # Extract the model-specific suffix: modeling_foo_bar.py -> foo_bar
    fname = file_path.name
    for prefix in ("modeling_", "modular_"):
        if fname.startswith(prefix):
            suffix = fname[len(prefix) :]  # e.g. "foo_bar.py"
            exact = model_dir / f"configuration_{suffix}"
            if exact.exists():
                return exact
            break

    # Fallback: pick any configuration file (single-config directories)
    candidates = sorted(model_dir.glob("configuration_*.py"))
    return candidates[0] if candidates else None


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    violations: list[Violation] = []

    # Only check modeling_*.py and modular_*.py files
    fname = file_path.name
    if not (fname.startswith("modeling_") or fname.startswith("modular_")):
        return violations

    # Collect all classes with non-empty _tied_weights_keys
    classes_with_tied_keys: list[ast.ClassDef] = []
    for node in iter_pretrained_classes(tree, source_lines, RULE_ID):
        assignments = _get_class_assignments(node)
        tied_keys = assignments.get("_tied_weights_keys")
        if tied_keys is not None and _is_non_empty_collection(tied_keys):
            classes_with_tied_keys.append(node)

    if not classes_with_tied_keys:
        return violations

    # Check the corresponding config file
    config_path = _find_config_file(file_path)
    if config_path is None:
        for node in classes_with_tied_keys:
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: {node.name} defines _tied_weights_keys but no configuration file "
                        f"was found in {file_path.parent}."
                    ),
                )
            )
        return violations

    if _config_has_tie_word_embeddings(config_path):
        return violations

    # Config exists but lacks tie_word_embeddings
    for node in classes_with_tied_keys:
        violations.append(
            Violation(
                file_path=file_path,
                line_number=node.lineno,
                message=(
                    f"{RULE_ID}: {node.name} defines _tied_weights_keys but {config_path.name} "
                    f"does not declare tie_word_embeddings. Add 'tie_word_embeddings: bool = True' "
                    f"to the config class."
                ),
            )
        )

    return violations
