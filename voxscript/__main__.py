"""Allow ``python -m voxscript``."""

import sys

from voxscript.cli import main

if __name__ == "__main__":
    sys.exit(main())
