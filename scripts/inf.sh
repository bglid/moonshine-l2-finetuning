#!/usr/bin/bash

set -euo pipefail

# running inference for baseline
uv run python -m scripts.evaluate \
	--model UsefulSensors/moonshine-tiny \
	\
	--dataset ./data/l2arctic_split \
	--split test \
	--audio-column audio \
	--text-column text \
	--device cuda \
	--fp16 # --dataset KoelLabs/L2Arctic \
