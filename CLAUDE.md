# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with documentation in this repository.

# Portico Documentation

This repository contains the MkDocs documentation site for the Portico framework. The documentation is built with MkDocs Material and includes comprehensive guides for ports, kits, and adapters.

## Documentation Standards

### Writing Style

**NEVER use emojis in markdown titles/headers**. This applies to all heading levels (#, ##, ###, etc.).

```markdown
❌ BAD - Emoji in title:
### 1. ✅ Keep Handlers Stateless

✅ GOOD - No emoji in title:
### 1. Keep Handlers Stateless
```

Emojis are acceptable in:
- Code comments within code blocks (e.g., `# ✅ GOOD - Handler is stateless`)
- Regular paragraph text (if contextually appropriate)
- Example outputs or demonstrations

**Rationale**: Markdown titles/headers should be clean and professional. Emojis can cause rendering issues, accessibility problems, and make the documentation less searchable and parseable.

### Markdown Formatting Rules

**ALWAYS add a blank line before lists**. Markdown requires a blank line before bullet lists and numbered lists to render them properly.

```markdown
❌ BAD - No blank line before list:
We welcome suggestions for:
- New demo applications
- Improvements to existing demos

✅ GOOD - Blank line before list:
We welcome suggestions for:

- New demo applications
- Improvements to existing demos
```

**ALWAYS add a blank line before nested lists**.

```markdown
❌ BAD - No blank line before nested list:
2. Create a new issue with:
   - Demo name
   - Steps to reproduce

✅ GOOD - Blank line before nested list:
2. Create a new issue with:

   - Demo name
   - Steps to reproduce
```

**Rationale**: Without the blank line, MkDocs will render the list items as regular paragraph text instead of formatted bullet points. This is a common Markdown parser behavior that affects readability.

### Documentation Structure

Each port documentation file should follow this template:

1. **Overview** - Purpose, domain, capabilities, port type, when to use
2. **Domain Models** (if applicable) - Complete field tables
3. **Enumerations** (if applicable) - All enum values with descriptions
4. **Port Interfaces** - Detailed method documentation with signatures
5. **Common Patterns** - 6-7 code examples showing typical usage
6. **Best Practices** - 7 items with good/bad examples
7. **FAQs** - 8 comprehensive questions and answers
8. **Related Ports** - Links to related documentation
9. **Architecture Notes** - How the port fits into hexagonal architecture

### Code Examples

- All code examples must be valid Python 3.13+ syntax
- Use type hints consistently
- Show both good (✅ GOOD) and bad (❌ BAD) examples in Best Practices
- Include complete, runnable examples where possible
- Add comments explaining non-obvious behavior

### Building Documentation

```bash
# Serve docs locally with live reload
poetry run mkdocs serve     # http://127.0.0.1:8000

# Build static site
poetry run mkdocs build     # Output to site/

# Build with strict mode (treat warnings as errors)
poetry run mkdocs build --strict

# Validate documentation coverage
poetry run python scripts/validate_docs_coverage.py

# Test code examples in docs
poetry run pytest docs_tests/
```

### Before Committing

Always run these checks before committing documentation changes:

1. **Build with strict mode**: `poetry run mkdocs build --strict` must show 0 errors
2. **Preview locally**: `poetry run mkdocs serve` and manually check the pages
3. **Verify navigation**: Ensure new pages are added to `mkdocs.yml` in alphabetical order
4. **Check links**: Verify all internal links work correctly
5. **No emojis in titles**: Grep for emojis in headers: `grep -rn "^###.*✅" docs/` should return nothing
6. **Lists have blank lines**: Verify all lists have a blank line before them (prevents rendering as text)

### File Organization

```
docs/
├── index.md                # Landing page
├── philosophy.md           # Architecture principles
├── ports/                  # Port interfaces (ABC/Protocol)
│   ├── index.md
│   └── *.md               # One file per port, alphabetically named
├── kits/                   # Service implementation guides
│   ├── index.md
│   └── *.md
├── adapters/               # Adapter implementation details
│   ├── index.md
│   └── *.md
└── stylesheets/           # Custom CSS
```

### Navigation (mkdocs.yml)

When adding new documentation:

1. Add the file to the appropriate section (Ports, Kits, or Adapters)
2. Use a descriptive title (e.g., "Job Domain Models" not just "Job")
3. Maintain alphabetical order within each section
4. Use consistent naming: `portname.md` for ports, `kitname.md` for kits

Example:
```yaml
nav:
  - Ports:
    - Overview: ports/index.md
    - Audit Port: ports/audit.md
    - Cache Port: ports/cache.md
    - Job Domain Models: ports/job.md
    - Job Creator Port: ports/job_creator.md
```

## Common Tasks

### Adding New Port Documentation

1. Create `docs/ports/portname.md` following the template structure
2. Add entry to `mkdocs.yml` under `Ports:` in alphabetical order
3. Build with `poetry run mkdocs build --strict`
4. Preview with `poetry run mkdocs serve`
5. Create a branch named `docs/add-portname-port-documentation`
6. Commit and create PR with detailed summary

### Updating Existing Documentation

1. Make changes to the relevant `.md` file
2. Build with strict mode to catch any errors
3. Preview changes locally
4. Commit with descriptive message explaining what was updated

### Cross-Referencing

Use relative links for internal documentation:

```markdown
See the [Job Handler Port](job_handler.md) for details on processing jobs.

For domain models, refer to [Job Domain Models](job.md#domain-models).
```

### Best Practices Section Format

Each best practice should:
- Have a numbered heading without emoji (e.g., `### 1. Keep Handlers Stateless`)
- Start with a brief explanation
- Include a code example showing the good approach (with `# ✅ GOOD` comment)
- Include a code example showing the bad approach (with `# ❌ BAD` comment)
- Explain why the good approach is better

## Tone and Style

- Professional and technical
- Direct and concise
- No marketing language or unnecessary superlatives
- Focus on facts and practical guidance
- Use active voice
- Assume reader is a competent Python developer

## Resources

- **MkDocs**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/
- **Main Library**: ../portico/
- **Examples**: ../portico-examples/
