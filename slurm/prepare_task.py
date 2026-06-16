#!/usr/bin/env python3
"""Set up one array task and print MAIN_ARGS for run.sbatch (not user-facing)."""

import sys

from slurm.experiments import EXPERIMENTS
from slurm.grid import prepare_task


def main() -> None:
    prepare_task(EXPERIMENTS[sys.argv[1]], int(sys.argv[2]))


if __name__ == "__main__":
    main()
