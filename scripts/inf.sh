#!/usr/bin/bash

set -euo pipefail

# running inference for baseline
python evaluate.py \
	--model UsefulSensors/moonshine-tiny \
	--dataset KoelLabs/L2Arctic \
	--split test \
	--audio-column audio \
	--text-column text \
	--device cuda \
	--fp16
