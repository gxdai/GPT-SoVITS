#!/bin/bash


# Check if exactly one argument is passed
if [ $# -ne 7 ]; then
    echo "Usage: $0 <gpt_model>  <sovits_model>  <ref_audio>  <target_text>  <target_language>  <ref_language>  <output_path>"
    exit 1
fi
python GPT_SoVITS/inference_cli.py --gpt_model $1 --sovits_model $2 --ref_audio $3 \
    --target_text $4 --target_language $5 --ref_language $6 --output_path $7\
    --ref_text ""


## Example cmd: GPT_SoVITS/inference_cli.py --gpt_model ./GPT_weights_v2/xxx-e15.ckpt --sovits_model ./SoVITS_weights_v2/xxx_e8_s384.pth --ref_audio ./elon_musk/opt_vocal/vocal_slice_001.mp3_10.flac --target_text "This is a great country, I love it. Tesla will produce more cars" --target_language  英文 --ref_language 英文 --output_path ./tmp --ref_text ""