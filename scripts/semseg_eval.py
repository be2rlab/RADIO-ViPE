"""Entrypoint for semantic segmentation evaluation."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semseg.semseg_eval import main


if __name__ == "__main__":
    main()
