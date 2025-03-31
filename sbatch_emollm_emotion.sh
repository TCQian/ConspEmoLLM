#!/bin/bash

#SBATCH --job-name=EmoLLM_emotion   # Job name
#SBATCH --time=0:30:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications
#SBATCH --mem=16G                        # Request 16GB of memory

source ~/.bashrc
conda activate conspemollm

export ABS_PATH="."

model_name_or_path=$ABS_PATH/model/Emollama-chat-7b
infer_file=$ABS_PATH/data/loco_test_emollm_emotion.json
predict_file=$ABS_PATH/predicts/loco_predict_emollm_emotion.json


python src/inference.py \
    --model_name_or_path $model_name_or_path \
    --infer_file $infer_file \
    --predict_file $predict_file \
    --batch_size 4 \
    --seed 123 \
    --llama
