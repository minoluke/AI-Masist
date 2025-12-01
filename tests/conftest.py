"""pytest configuration"""
import sys
from pathlib import Path

# Add tests directory to path for fixtures import
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

# Add project root to path
project_root = tests_dir.parent
sys.path.insert(0, str(project_root))
