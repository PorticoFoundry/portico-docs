# Documentation Scripts

This directory contains scripts for validating and maintaining the Portico documentation.

## validate_markdown_style.py

Validates markdown files for style consistency according to the Portico documentation standards.

### Checks Performed

1. **No emojis in markdown headers** - Headers (# ## ### etc.) should not contain emojis
   - Emojis are allowed in code comments within code blocks
   - Example violation: `## Setup ✅` (should be `## Setup`)

2. **Blank lines before lists** - List items must have a blank line before them
   - Prevents rendering issues in MkDocs
   - Example violation:
     ```markdown
     Some text here.
     - List item  # ❌ No blank line before list
     ```

   - Correct:
     ```markdown
     Some text here.

     - List item  # ✅ Blank line before list
     ```

3. **No trailing whitespace** - Lines should not end with spaces or tabs
   - Keeps files clean and prevents diff noise

4. **Files end with newline** - All files should end with a newline character
   - Standard POSIX requirement

### Usage

Run manually on specific files:

```bash
python3 scripts/validate_markdown_style.py docs/kits/cache.md
python3 scripts/validate_markdown_style.py docs/kits/*.md
python3 scripts/validate_markdown_style.py docs/**/*.md
```

Run on all markdown files:

```bash
find docs -name "*.md" -exec python3 scripts/validate_markdown_style.py {} +
```

### Exit Codes

- `0` - All checks passed
- `1` - Style violations found

## Pre-Commit Hook

A git pre-commit hook automatically runs markdown validation on staged files before each commit.

### Installation

The hook is already installed at `.git/hooks/pre-commit`. To verify:

```bash
ls -la .git/hooks/pre-commit
```

### How It Works

When you run `git commit`:

1. The hook identifies all staged markdown files
2. Runs `validate_markdown_style.py` on those files
3. Optionally runs `mkdocs build --strict` to verify docs build
4. If any check fails, the commit is blocked with error messages
5. If all checks pass, the commit proceeds

### Bypassing the Hook

If you need to commit despite validation failures (not recommended):

```bash
git commit --no-verify
```

### Testing the Hook

Stage a markdown file with intentional violations:

```bash
# Create test file
cat > test_style.md << 'EOF'
# Header with emoji ✅

Some text.
- List without blank line
EOF

# Stage it
git add test_style.md

# Try to commit
git commit -m "test commit"
# Should fail with style violations

# Fix the file
cat > test_style.md << 'EOF'
# Header without emoji

Some text.

- List with blank line
EOF

# Stage and commit again
git add test_style.md
git commit -m "test commit"
# Should succeed

# Clean up
git reset HEAD~1
rm test_style.md
```

## validate_docs_coverage.py

Validates that all ports, kits, and adapters have corresponding documentation.

### Usage

```bash
poetry run python scripts/validate_docs_coverage.py
```

This checks:

- All ports in `portico/portico/ports/` have docs in `docs/ports/`
- All kits in `portico/portico/kits/` have docs in `docs/kits/`
- All adapters in `portico/portico/adapters/` have docs in `docs/adapters/`
- All documented items exist in the codebase

## CI Integration

The markdown style validation is integrated into CI:

```yaml
# .github/workflows/docs.yml
- name: Validate markdown style
  run: |
    find docs -name "*.md" -exec python3 scripts/validate_markdown_style.py {} +

- name: Build docs
  run: poetry run mkdocs build --strict
```

## Troubleshooting

### Hook not running

If the pre-commit hook isn't running, check:

1. Hook is executable: `chmod +x .git/hooks/pre-commit`
2. Hook file exists: `ls -la .git/hooks/pre-commit`
3. You're in the correct git repository

### Python not found

If you get "python3: command not found":

1. Ensure Python 3 is installed: `python3 --version`
2. Update hook to use correct Python path: `which python3`

### False positives

If the script incorrectly flags valid markdown:

1. Check the specific violation reported
2. Verify it's not a nested list (which doesn't need a blank line)
3. Verify emojis are in code blocks, not headers
4. Report issues to the team

## Adding New Checks

To add a new style check:

1. Add check function to `validate_markdown_style.py`:
   ```python
   def check_new_rule(lines: List[str]) -> List[Tuple[int, str]]:
       """Check for new rule violation."""
       violations = []
       # Implement check logic
       return violations
   ```

2. Call it in `validate_markdown_file()`:
   ```python
   new_violations = check_new_rule(lines)
   if new_violations:
       errors.append("\n  New rule violations:")
       # Format errors
   ```

3. Update this README with documentation
4. Test with known good and bad examples
