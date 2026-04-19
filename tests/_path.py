"""Auto-inject project root into sys.path so tests can import root-level
modules (pipeline, search, wikidata, etc.) without installing the project
as a package. Imported by every test file."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
