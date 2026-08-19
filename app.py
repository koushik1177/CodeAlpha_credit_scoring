"""
Main Application Entrypoint.

Allows running `streamlit run app.py` directly from VS Code root directory.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
DATABASE_DIR = ROOT_DIR / "database"
FRONTEND_DIR = ROOT_DIR / "frontend"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(DATABASE_DIR), str(FRONTEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from frontend.app import main

if __name__ == "__main__":
    main()
