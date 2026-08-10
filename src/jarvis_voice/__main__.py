"""Module entry point kept safe for multiprocessing spawn imports."""

import multiprocessing

from .cli import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
