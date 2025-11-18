#!/bin/bash
python src/inference_path.py \
    --input_path examples \
    --negative_prompt "heavy rain, light rain, drizzle, sprinkle, shower, raindrops, rainstorm, streak" \
    --skip_value 40 \
    --editing_guidance_scale 15