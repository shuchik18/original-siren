"""Command-line wrapper for the Siren compiler."""

from llcsiren.src.compiler import *  # noqa: F401,F403
from llcsiren.src.compiler import main


if __name__ == "__main__":
  raise SystemExit(main())
