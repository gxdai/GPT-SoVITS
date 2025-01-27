#!/bin/bash

# step 1: split vocal and instr
python tools/uvr5/webui_cli.py

# step 2: slice vocal

# input audio dir: output/opt_vocal
# output sliced dir: output/slice_opt_tmp
python tools/slice_audio.py output/opt_vocal output/slicer_opt -34 4000 300 10 500 0.9 0.25 3 4

# Step 3: denoise audio

python tools/cmd-denoise.py -i output/slicer_opt -o output/denoise_opt -p float32

# Step 4: ASR

python tools/asr/funasr_asr.py -i "output/denoise_opt" -o "output/asr_opt" -s large -l zh -p float32

# Step 5: proof read skip...

# step 6: prepare datasets
python prepare_dataset.py

# Step 7: Sovits training
# models are saved under `SoVITS_weights_v2`

python train_sovits.py

# Step 8: GPT training
# models are saved under `GPT_weights_v2`

python train_gpt.py