import sys
from pathlib import Path

# Make `import self_healing` work when pytest is run from the repo root
# without an editable install (CI installs only requirements*.txt).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
