#!/usr/bin/env python3
"""Validate that documentation exists for all ports, adapters, and kits.

This script ensures that every component in the codebase has corresponding
documentation, preventing undocumented features.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set


class ComponentFinder(ast.NodeVisitor):
    """Find adapter/port classes in Python source."""

    def __init__(self) -> None:
        self.components: Set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Skip internal implementation classes that don't need dedicated docs
        # These are documented in aggregate or as implementation details
        skip_patterns = [
            "SqlAlchemy.*Repository",  # Repository impls documented in aggregate
            ".*MetadataSource",  # Internal metadata classes
            "DefaultMetadataRegistry",  # Internal registry
            "DatabaseFeatureRegistry",  # Internal registry
            "MemorySettingsRegistry",  # Internal registry
            "MemoryPromptRegistry",  # Internal registry
            "DictProvider",  # Internal provider
            "PostgresAdapter",  # Thin wrapper, documented in PostgreSQL guide
            "GroupAwareFileStorageService",  # Internal service
            "OrganizationFileStorageService",  # Internal service
        ]
        import re

        for pattern in skip_patterns:
            if re.match(pattern, node.name):
                self.generic_visit(node)
                return

        # Check if class inherits from Adapter, Repository, Registry, etc.
        for base in node.bases:
            base_name = ast.unparse(base)
            if any(
                suffix in base_name
                for suffix in [
                    "Adapter",
                    "Repository",
                    "Registry",
                    "Provider",
                    "Service",
                ]
            ):
                self.components.add(node.name)
        self.generic_visit(node)


def find_code_components(base_path: Path, component_type: str) -> Dict[str, List[str]]:
    """Find all components of a given type (ports, adapters, kits)."""
    components: Dict[str, List[str]] = {}

    search_path = base_path / "portico" / component_type
    if not search_path.exists():
        return components

    for py_file in search_path.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue

        try:
            with open(py_file, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            finder = ComponentFinder()
            finder.visit(tree)

            if finder.components:
                # Get relative module path
                rel_path = py_file.relative_to(search_path.parent)
                module = str(rel_path.with_suffix("")).replace("/", ".")
                components[module] = sorted(finder.components)

        except SyntaxError:
            print(f"⚠️  Syntax error in {py_file}, skipping")
            continue

    return components


def find_documented_components(docs_path: Path, component_type: str) -> Set[str]:
    """Find all components that have documentation."""
    documented = set()

    doc_dir = docs_path / component_type
    if not doc_dir.exists():
        return documented

    for md_file in doc_dir.rglob("*.md"):
        if md_file.name == "index.md":
            continue

        # Extract component name from filename
        # Add both the stem and the full relative path (without extension)
        documented.add(md_file.stem)

        # Also add the relative path from the component type directory
        # e.g., cache/memory.md -> cache/memory
        rel_path = md_file.relative_to(doc_dir).with_suffix("")
        documented.add(str(rel_path).replace("/", "."))

        # Scan file content for class names mentioned
        try:
            content = md_file.read_text(encoding="utf-8")
            # Look for class names mentioned with backticks or code blocks
            # This handles aggregate docs like audit.md that document multiple classes
            for line in content.splitlines():
                # Find class-like names (CamelCase words potentially in backticks)
                import re

                # Match patterns like `ClassName` or ClassName at start of line/header
                patterns = [
                    r"`([A-Z][a-zA-Z0-9]+(?:Adapter|Repository|Provider|Registry|Service))`",
                    r"^##?\s+([A-Z][a-zA-Z0-9]+(?:Adapter|Repository|Provider|Registry|Service))",
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        # Add the full class name in lowercase
                        documented.add(match.lower())
                        # Also add without common suffixes for easier matching
                        base = match.lower()
                        for suffix in [
                            "adapter",
                            "provider",
                            "repository",
                            "registry",
                            "service",
                        ]:
                            base = base.replace(suffix, "")
                        documented.add(base)
        except Exception:
            pass  # If we can't read the file, skip it

    return documented


def check_component_type(
    base_path: Path, docs_path: Path, component_type: str
) -> tuple[bool, List[str]]:
    """Check documentation coverage for a component type."""
    print(f"\n{'=' * 60}")
    print(f"Checking {component_type.upper()} documentation coverage")
    print(f"{'=' * 60}")

    code_components = find_code_components(base_path, component_type)
    documented = find_documented_components(docs_path, component_type)

    missing = []
    for module, classes in code_components.items():
        for class_name in classes:
            # Try multiple naming conventions to find documentation
            # Extract module path after "adapters."
            module_parts = module.split(".")
            if len(module_parts) > 1:
                # e.g., "adapters.cache.memory_cache" -> "cache.memory"
                submodule = ".".join(module_parts[1:])
            else:
                submodule = ""

            # Generate possible doc names
            possible_names = set()

            # 1. Just the class name in lowercase without suffixes
            base_name = class_name.lower()
            for suffix in ["adapter", "provider", "repository", "registry", "service"]:
                base_name = base_name.replace(suffix, "")
            possible_names.add(base_name)

            # 2. Module name (e.g., "memory" from "memory_cache")
            if submodule:
                module_name = submodule.split(".")[-1].replace("_", "")
                possible_names.add(module_name)

                # 3. Full submodule path (e.g., "cache.memory")
                submodule_clean = (
                    submodule.replace("_cache", "")
                    .replace("_queue", "")
                    .replace("_audit", "")
                    .replace("_storage", "")
                )
                submodule_clean = (
                    submodule_clean.replace("_provider", "")
                    .replace("_registry", "")
                    .replace("_repository", "")
                )
                possible_names.add(submodule_clean)

            # Check if any of the possible names are documented
            if not any(name in documented for name in possible_names):
                missing.append(f"{module}.{class_name}")

    if missing:
        print(f"❌ Missing documentation for {len(missing)} component(s):")
        for component in sorted(missing):
            print(f"   - {component}")
        return False, missing
    print(f"✅ All {component_type} documented ({len(code_components)} modules)")
    return True, []


def main() -> int:
    """Main validation logic."""
    base_path = Path(__file__).parent.parent
    docs_path = base_path / "docs"

    print("\n" + "=" * 60)
    print("PORTICO DOCUMENTATION COVERAGE VALIDATION")
    print("=" * 60)

    all_ok = True
    all_missing = []

    # Check each component type
    for component_type in ["ports", "adapters", "kits"]:
        ok, missing = check_component_type(base_path, docs_path, component_type)
        all_ok = all_ok and ok
        all_missing.extend(missing)

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ SUCCESS: All components documented")
        print("=" * 60)
        return 0
    print(f"❌ FAILURE: {len(all_missing)} undocumented component(s)")
    print("=" * 60)
    print("\nPlease add documentation for the missing components.")
    print("See docs/templates/ for documentation templates.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
