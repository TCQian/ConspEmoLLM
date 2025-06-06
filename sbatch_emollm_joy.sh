#!/bin/bash

#SBATCH --job-name=EmoLLM_joy   # Job name
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate conspemollm

export ABS_PATH="."

model_name_or_path=$ABS_PATH/model/Emollama-chat-7b
infer_file=$ABS_PATH/data/loco_test_emollm_joy_full.json
predict_file=$ABS_PATH/predicts/loco_predict_emollm_joy_full.json


python src/inference.py \
    --model_name_or_path $model_name_or_path \
    --infer_file $infer_file \
    --predict_file $predict_file \
    --batch_size 2 \
    --seed 123 \
    --load_type fp32 \
    --llama \
