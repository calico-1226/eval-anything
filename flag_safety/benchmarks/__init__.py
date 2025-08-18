"""
Benchmarks package initialization.

This module initializes the benchmarks package and handles registration of all evaluators.
"""

# automatically import all benchmarks
import os
import importlib
import logging

# First import the registry
from flag_safety.benchmarks.registry import BenchmarkRegistry

# use `LAZY_IMPORT` in `__init__.py` of each sub-package to import the evaluator class
logger = logging.getLogger(__name__)


def _discover_and_import_benchmarks() -> list:
    """Discover subpackages and import evaluators declared via LAZY_IMPORT.

    Each benchmark subpackage should define a module-level variable `LAZY_IMPORT`
    as a list of tuples: [(module_path, attribute_name)]. We import the module
    and access the attribute to trigger class registration via decorators.
    """

    discovered_exports = []
    base_dir = os.path.dirname(__file__)
    base_pkg = __name__  # e.g., "flag_safety.benchmarks"

    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.startswith("__") or entry.startswith("."):
            continue
        # ensure it's a package
        if not os.path.exists(os.path.join(entry_path, "__init__.py")):
            continue

        subpkg_name = f"{base_pkg}.{entry}"
        try:
            subpkg = importlib.import_module(subpkg_name)
        except Exception as exc:
            logger.warning(f"Skipping subpackage {subpkg_name}: {exc}")
            continue

        spec = getattr(subpkg, "__all__", None)
        if spec:
            discovered_exports.extend(spec)

    return discovered_exports


_exported_classes = _discover_and_import_benchmarks()

__all__ = [
    "BenchmarkRegistry",
    *_exported_classes,
]

