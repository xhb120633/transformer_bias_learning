"""Make the standalone code package's flat research scripts importable in tests."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(folder))
