#!/bin/bash

# Check if exactly one argument is passed
if [ $# -ne 3 ]; then
    echo "Usage: $0 <audio_root_directory> <output_directory> <language: zh, cn>"
    exit 1
fi

audio_root_directory=$1
output_dir=$2
language=$3

# Check if the directory exists
if [ ! -d "$output_dir" ]; then
    echo "Directory does not exist. Creating: $output_dir"
    mkdir -p "$output_dir"
else
    echo "Directory already exists: $output_dir"
fi

echo "##################################################################"
echo "All the results will saved under $output_dir"
echo "##################################################################"

sleep 2

# step 1: split vocal and instr
echo "Split vocal and instruments from the audio>>>>"
sleep 1
python tools/uvr5/webui_cli.py --dr-wav-input ${audio_root_directory} \
    --opt-vocal-root ./${output_dir}/opt_vocal \
    --opt-instr-root ./${output_dir}/opt_instr

# step 2: slice vocal

# input audio dir: output/opt_vocal
# output sliced dir: output/slice_opt_tmp

echo "Slice audio into small pieces>>>>"
sleep 1
python tools/slice_audio.py ./${output_dir}/opt_vocal ${output_dir}/opt_slicer -34 4000 300 10 500 0.9 0.25 3 4

# Step 3: denoise audio
echo "Denoising audios>>>>"
sleep 1
python tools/cmd-denoise.py -i ${output_dir}/opt_slicer -o ${output_dir}/opt_denoise -p float32

# Step 4: ASR
## For chines use this model 

if [ "${language}" = "zh" ]; then
    python tools/asr/funasr_asr.py -i "${output_dir}/opt_denoise" -o "${output_dir}/opt_asr" -s large -l zh -p float32
else
    python tools/asr/fasterwhisper_asr.py -i ${output_dir}/opt_denoise -o ${output_dir}/opt_asr -s large-v3-local -l auto -p int8
fi

# step 6: prepare datasets
python prepare_dataset.py -e ${output_dir}

# Step 7: Sovits training
# models are saved under `SoVITS_weights_v2`

python train_sovits.py -e ${output_dir}

# Step 8: GPT training
# models are saved under `GPT_weights_v2`

python train_gpt.py -e ${output_dir}

echo "All the models are saved under <${output_dir}/SoVITS_weights_v2> and <${output_dir}/GPT_weights_v2>"