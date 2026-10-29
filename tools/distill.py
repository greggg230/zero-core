"""python tools/distill.py ...: zero.tools.distill (docstring below) for the core's own toy runs.
Each game repo has the same script for its package, running its pinned core."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
from zero.tools.distill import *  # noqa: E402,F401,F403
from zero.tools.distill import __doc__, cli, main  # noqa: E402,F401

if __name__ == '__main__':
    cli()
