"""PyInstaller entry point (the frozen app starts here)."""

import sys

from chibi.app import main

if __name__ == "__main__":
    sys.exit(main())
