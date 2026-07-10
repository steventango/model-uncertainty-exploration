#!/bin/bash
rsync -avz --progress vulcan:/home/stang5/scratch/model-uncertainty-exploration/runs/blr_enn runs/
uv run python scripts/plot_learning_curves.py --root runs/blr_enn
