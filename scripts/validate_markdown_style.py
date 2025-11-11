#!/usr/bin/env python3
"""Validate markdown documentation style.

This script checks markdown files for style violations:
1. No emojis in markdown headers (# ## ### etc.)
2. Blank lines before lists (- or * items)
3. No trailing whitespace
4. Files end with newline

Usage:
    python scripts/validate_markdown_style.py [files...]
    python scripts/validate_markdown_style.py docs/kits/*.md

Exit codes:
    0 - All checks passed
    1 - Style violations found
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Common emoji patterns to detect
EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F'  # Emoticons
    r'\U0001F300-\U0001F5FF'  # Symbols & Pictographs
    r'\U0001F680-\U0001F6FF'  # Transport & Map
    r'\U0001F1E0-\U0001F1FF'  # Flags
    r'\U00002702-\U000027B0'  # Dingbats
    r'\U000024C2-\U0001F251'  # Enclosed chars
    r'\u2705\u274C\u2714\u2716'  # Check marks, X marks
    r']+'
)


def check_emoji_in_headers(lines: List[str]) -> List[Tuple[int, str]]:
    """Check for emojis in markdown headers (outside code blocks).

    Args:
        lines: List of file lines

    Returns:
        List of (line_number, line_text) tuples with violations
    """
    violations = []
    in_code_block = False

    for i, line in enumerate(lines, 1):
        # Track code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue

        # Check for markdown headers (not in code blocks)
        if not in_code_block and re.match(r'^#{1,6}\s+', line):
            if EMOJI_PATTERN.search(line):
                violations.append((i, line.strip()))

    return violations


def check_blank_before_lists(lines: List[str]) -> List[Tuple[int, str]]:
    """Check for blank lines before list items.

    Args:
        lines: List of file lines

    Returns:
        List of (line_number, line_text) tuples with violations
    """
    violations = []
    in_code_block = False

    for i, line in enumerate(lines, 1):
        # Track code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue

        # Skip if in code block
        if in_code_block:
            continue

        # Check if line is a list item
        if re.match(r'^[\s]*[-*]\s', line):
            # Check if previous line exists and is not blank
            if i > 1:
                prev_line = lines[i-2]  # i is 1-indexed, list is 0-indexed

                # Violation if previous line is not blank and not a list item
                if prev_line.strip() and not re.match(r'^[\s]*[-*]\s', prev_line):
                    # Also check it's not a nested list (indented more than parent)
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    curr_indent = len(line) - len(line.lstrip())

                    # If current line is not more indented, it needs blank line
                    if curr_indent <= prev_indent:
                        violations.append((i, line.strip()[:60]))

    return violations


def check_trailing_whitespace(lines: List[str]) -> List[Tuple[int, str]]:
    """Check for trailing whitespace on lines.

    Args:
        lines: List of file lines

    Returns:
        List of (line_number, line_text) tuples with violations
    """
    violations = []

    for i, line in enumerate(lines, 1):
        # Check if line ends with whitespace (but not just newline)
        if line.rstrip('\n') != line.rstrip():
            violations.append((i, line.rstrip('\n')[:60] + ' [trailing whitespace]'))

    return violations


def check_file_ends_with_newline(lines: List[str], filepath: Path) -> bool:
    """Check if file ends with newline.

    Args:
        lines: List of file lines
        filepath: Path to file (for reading raw content)

    Returns:
        True if file ends with newline, False otherwise
    """
    if not lines:
        return True

    # Read raw content to check final character
    with open(filepath, 'rb') as f:
        content = f.read()
        return content.endswith(b'\n')


def validate_markdown_file(filepath: Path) -> Tuple[bool, str]:
    """Validate a single markdown file.

    Args:
        filepath: Path to markdown file

    Returns:
        Tuple of (passed, error_message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return False, f"Error reading file: {e}"

    errors = []

    # Check 1: No emojis in headers
    emoji_violations = check_emoji_in_headers(lines)
    if emoji_violations:
        errors.append("\n  Emojis in markdown headers:")
        for line_no, text in emoji_violations:
            errors.append(f"    Line {line_no}: {text}")

    # Check 2: Blank lines before lists
    list_violations = check_blank_before_lists(lines)
    if list_violations:
        errors.append("\n  Lists without blank line before them:")
        for line_no, text in list_violations:
            errors.append(f"    Line {line_no}: {text}")

    # Check 3: No trailing whitespace
    whitespace_violations = check_trailing_whitespace(lines)
    if whitespace_violations:
        errors.append("\n  Trailing whitespace:")
        for line_no, text in whitespace_violations[:5]:  # Limit to 5
            errors.append(f"    Line {line_no}: {text}")
        if len(whitespace_violations) > 5:
            errors.append(f"    ... and {len(whitespace_violations) - 5} more")

    # Check 4: File ends with newline
    if not check_file_ends_with_newline(lines, filepath):
        errors.append("\n  File does not end with newline")

    if errors:
        return False, "\n".join(errors)

    return True, ""


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_markdown_style.py <file1.md> [file2.md ...]")
        print("\nChecks markdown files for style violations:")
        print("  - No emojis in markdown headers")
        print("  - Blank lines before lists")
        print("  - No trailing whitespace")
        print("  - Files end with newline")
        sys.exit(1)

    files_to_check = [Path(f) for f in sys.argv[1:]]

    # Filter to only markdown files
    markdown_files = [f for f in files_to_check if f.suffix == '.md' and f.exists()]

    if not markdown_files:
        print("No markdown files to check.")
        sys.exit(0)

    print(f"Checking {len(markdown_files)} markdown file(s)...\n")

    all_passed = True
    failed_files = []

    for filepath in markdown_files:
        passed, error_msg = validate_markdown_file(filepath)

        if passed:
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath}")
            print(error_msg)
            print()
            all_passed = False
            failed_files.append(filepath)

    print()
    if all_passed:
        print(f"✓ All {len(markdown_files)} file(s) passed style checks")
        sys.exit(0)
    else:
        print(f"✗ {len(failed_files)} file(s) failed style checks:")
        for f in failed_files:
            print(f"  - {f}")
        print("\nStyle Guide:")
        print("  - Don't use emojis in markdown headers (# ## ### etc.)")
        print("  - Use emojis only in code comments within code blocks")
        print("  - Add blank line before list items")
        print("  - Remove trailing whitespace")
        print("  - Ensure files end with newline")
        sys.exit(1)


if __name__ == "__main__":
    main()
