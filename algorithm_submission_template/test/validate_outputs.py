#!/usr/bin/env python3
"""
Validate outputs produced by do_test_run.sh.

Checks that expected files exist and match the platform contract
(plain JSON string for interf0; bare step array for interf1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

INTERF0_OUTPUT = OUTPUT_DIR / "interf0" / "visual-context-response.json"
INTERF1_OUTPUT = OUTPUT_DIR / "interf1" / "chain-of-thought.json"

CHAIN_STEP_KEYS = ("question", "answer", "next_question")


def _fail(message: str) -> None:
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def validate_interf0_output(path: Path) -> None:
    if not path.is_file():
        _fail(f"Interface 0 output not found: {path}")

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"Interface 0 output is not valid JSON ({path}): {exc}")

    if not isinstance(content, str):
        _fail(
            "Interface 0 output must be a plain JSON string "
            f"(got {type(content).__name__}). "
            f"Example: \"Yes\" or \"not_informative\""
        )

    if not content.strip():
        _fail("Interface 0 output string must not be empty.")


def validate_interf1_output(path: Path) -> None:
    if not path.is_file():
        _fail(f"Interface 1 output not found: {path}")

    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"Interface 1 output is not valid JSON ({path}): {exc}")

    if isinstance(content, dict):
        _fail(
            "Interface 1 output must be a bare JSON array, not an object. "
            "Do not wrap steps in {\"id\": ...} — the platform adds that."
        )

    if not isinstance(content, list):
        _fail(
            f"Interface 1 output must be a JSON array (got {type(content).__name__})."
        )

    if not content:
        _fail("Interface 1 chain-of-thought array must contain at least one step.")

    for index, step in enumerate(content):
        if not isinstance(step, dict):
            _fail(f"Step {index} must be a JSON object (got {type(step).__name__}).")

        missing = [key for key in CHAIN_STEP_KEYS if key not in step]
        if missing:
            _fail(f"Step {index} is missing required keys: {', '.join(missing)}")

        extra = [key for key in step if key not in CHAIN_STEP_KEYS]
        if extra:
            _fail(f"Step {index} has unexpected keys: {', '.join(extra)}")

        for key in CHAIN_STEP_KEYS:
            if not isinstance(step[key], str):
                _fail(f"Step {index} field '{key}' must be a string.")

    last_next = content[-1]["next_question"]
    if last_next != "":
        _fail(
            "Last chain-of-thought step must have \"next_question\": \"\" "
            f"(got {last_next!r})."
        )


def main() -> int:
    root = SCRIPT_DIR.parent
    print()
    print("=+= Validating output files and structure")
    validate_interf0_output(INTERF0_OUTPUT)
    print(f"=+= OK  {INTERF0_OUTPUT.relative_to(root)}")
    validate_interf1_output(INTERF1_OUTPUT)
    print(f"=+= OK  {INTERF1_OUTPUT.relative_to(root)}")
    print()
    print("=+= Output validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
