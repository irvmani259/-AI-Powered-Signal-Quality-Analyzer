# conftest.py — shared pytest configuration
import sys
from pathlib import Path

# Ensure src/ is on the path for all tests
sys.path.insert(0, str(Path(__file__).parent / "src"))
