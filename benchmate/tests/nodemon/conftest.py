import sys
from pathlib import Path

# Import the repository's benchmate, not a copy installed in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
