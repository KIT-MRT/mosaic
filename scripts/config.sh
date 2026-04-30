#!/usr/bin/env bash
# Hardware settings for experiment scripts. Edit to match your machine.

# Ray resource allocation per simulation — controls concurrency per GPU.
# Lower values run more simulations in parallel but use more VRAM in total.
#   0.05 → 20 sims/GPU
#   0.10 → 10 sims/GPU
GPUS_PER_SIM=0.05

# CPU threads available to Ray on this node.
THREADS=160
