import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
import argparse
import pandas as pd
import json
import os
import random
import numpy as np
import unicodedata


def seed_everything(seed=23):
    print("seed", seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def clean_unicode(text):
    return unicodedata.normalize("NFKC", text)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--llama", action="store_true")
parser.add_argument("--infer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--load_type", type=str, default="fp16")
args = parser.parse_args()

seed_everything(args.seed)

if args.ckpt_path is None or args.ckpt_path == "":
    args.ckpt_path = args.model_name_or_path

max_new_tokens = 1024
generation_config = dict(
    temperature=0.9,
    top_k=30,
    top_p=0.6,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens,
)

infer_data = pd.read_json(args.infer_file, lines=True)
instruction_list = infer_data.apply(
    lambda row: pd.Series(
        {
            "instruction": clean_unicode(
                f"Human: \n" + row["instruction"] + "\n\nAssistant:\n"
            )
        }
    ),
    axis=1,
)["instruction"].to_list()

if __name__ == "__main__":
    load_type = torch.float16 if args.load_type == "fp16" else torch.float32
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, device_map="auto"
        )
        model = PeftModel.from_pretrained(
            base_model, args.ckpt_path, torch_dtype=load_type
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=load_type,
            config=model_config,
            device_map="auto",
        )

    if device == torch.device("cpu"):
        model.float()

    model.eval()
    print("Load model successfully")
    batch_size = args.batch_size
    with open(args.predict_file, "w", encoding="utf-8") as write_f:
        for i in range(0, len(instruction_list), batch_size):
            batch_data = instruction_list[
                i : min(i + batch_size, len(instruction_list))
            ]
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            generation_output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generation_config
            )
            for j in range(generation_output.shape[0]):
                response = tokenizer.decode(
                    generation_output[j],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                data_one = {"output": response}
                if "output" in infer_data.iloc[i + j]:
                    ground_truth = infer_data.iloc[i + j]["output"]
                    pred = response.split("Assistant:", 1)
                    is_pred_correct = False
                    if len(pred) > 1:
                        is_pred_correct = ground_truth in pred[1].strip()

                    data_one["ground_truth"] = ground_truth
                    data_one["is_prediction_correct"] = is_pred_correct

                write_f.write(
                    json.dumps(data_one, indent=None, ensure_ascii=False) + "\n"
                )
