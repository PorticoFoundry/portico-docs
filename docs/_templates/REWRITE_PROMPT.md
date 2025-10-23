# Port Documentation Rewrite Prompt

Use this prompt with Claude to condense existing verbose port documentation into the shorter template format.

---

## Prompt for Claude

```
I need you to condense existing port documentation to be shorter while keeping all essential information. Follow this template structure and guidelines:

## GUIDELINES

### Length Target
- Target: 300-500 lines (down from 1000+)
- Focus on essential API reference and most common use cases

### What to Keep (Essential)
✅ Complete domain model field tables
✅ All method signatures with types
✅ 2-3 key methods with full documentation (parameters, returns, examples)
✅ At least one complete working example per model
✅ Top 5 best practices
✅ Top 3-5 most critical FAQs
✅ One integration example with kits

### What to Simplify or Remove
❌ Remove "Architecture Role" diagrams (same for all ports)
❌ Remove verbose "Usage Patterns" section (examples already in method docs)
❌ Condense "Integration with Kits" to one code example + link to kit docs
❌ Reduce 8-11 FAQs to top 3-5 most critical
❌ Reduce 10 best practices to top 5 key points
❌ Remove redundant examples (one per concept is enough)
❌ For non-key methods: just signature + one-line description

### Method Documentation Strategy
1. Identify the 2 most important/commonly-used methods
2. Document those 2 in full detail under "#### Key Methods"
3. All other methods go under "#### Other Methods" with just:
   - Method signature in code block
   - Single-line description

### Template Structure to Follow

```markdown
# {PORT_NAME} Port

## Overview
[Brief purpose, domain, key capabilities, when to use - keep concise]

## Domain Models
[Tables + one example per model - no verbose explanations]

## Enumerations
[Table of values if applicable]

## Port Interfaces

### {InterfaceName}

#### Key Methods
[2 most important methods with full docs + examples]

#### Other Methods
[All other methods: just signature + one line]

## Common Patterns
[1-2 most important patterns only - remove rest]

## Integration with Kits
[One code snippet + link to kit docs]

## Best Practices
[Top 5 as numbered list with brief ✅/❌ examples]

## FAQs
[Top 3-5 most critical questions only]
```

## YOUR TASK

Read the attached port documentation file and rewrite it following:

1. **Use the template structure above**
2. **Keep all field tables and method signatures** (complete API reference)
3. **Choose the 2 most important methods** for detailed documentation
4. **Collapse remaining methods** to signature + one-liner
5. **Keep only 1-2 usage patterns** (the most illustrative)
6. **Simplify integration** to one example + link
7. **Reduce best practices to top 5**
8. **Keep only 3-5 most critical FAQs**
9. **Use h5 headings (`#####`) for method names**
10. **Maintain all code examples that are kept** - just reduce quantity

## SELECTION CRITERIA

**For Key Methods**: Choose methods that are:
- Most frequently used in typical applications
- Cover the primary use case of the port
- Example: For audit port, `log_event()` and `search_events()` are most critical

**For Best Practices**: Keep practices that:
- Prevent security vulnerabilities
- Avoid common mistakes
- Are compliance-critical
- Have major performance impact

**For FAQs**: Keep questions about:
- Most common confusion points
- Critical security/compliance concerns
- Most asked implementation questions

## OUTPUT

Provide the complete rewritten documentation that:
- Follows the shorter template structure
- Is 300-500 lines (vs 1000+)
- Preserves all essential API information
- Removes redundancy and verbosity
- Keeps the most salient examples and explanations

---

[ATTACH THE EXISTING PORT DOCUMENTATION FILE HERE]
```

---

## Example Usage

1. Copy the prompt above
2. Attach the existing port documentation file (e.g., `docs/ports/audit.md`)
3. Send to Claude
4. Review the output for completeness
5. Replace the existing documentation

## Notes

- The rewritten docs should still be comprehensive API references
- Users should still be able to understand all capabilities
- Focus is on removing redundancy, not removing information
- Critical examples and best practices are preserved
