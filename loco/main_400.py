import pandas as pd
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../model/ConspEmoLLM-7b")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"


def truncate_and_check(text):
    # Encode original (no truncation)
    original_ids = tokenizer.encode(text, add_special_tokens=False)
    encoded = tokenizer(text, truncation=True, max_length=400, add_special_tokens=False)

    # Decode back
    truncated_text = tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )

    # Check if it was truncated
    was_truncated = len(encoded["input_ids"]) < len(original_ids)
    return truncated_text, was_truncated


loco_dataset = None
with open("LOCO.json", "r") as file:
    loco_dataset = json.loads(file.read())  # loco = [{...}, {...}, {...}, ...]

loco_dataset = dict([x["doc_id"], x] for x in loco_dataset)

f1 = pd.read_csv(
    "LOCO_covid_ANNOTATION.csv"
)  # doc id,subcorpus,seeds,date,annotator,conspiracy,relatedness
f2 = pd.read_csv(
    "LOCO_sandy.hook_ANNOTATION.csv"
)  # doc id,subcorpus,seeds,date,annotator,conspiracy,relatedness


# Process input required for ConspEmoLLM
processed_lst = []
for i, row in f1.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Determine if the text is a conspiracy theory. Classify it into one of the following two classes: 0. non-conspiracy. 1. conspiracy.\nText: {}\nClass:\n".format(
        loco_text
    )
    input = ""
    output = "1. conspiracy" if row["conspiracy"] == "Y" else "0. non-conspiracy"
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

for i, row in f2.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Determine if the text is a conspiracy theory. Classify it into one of the following two classes: 0. non-conspiracy. 1. conspiracy.\nText: {}\nClass:\n".format(
        loco_text
    )
    input = ""
    output = "1. conspiracy" if row["conspiracy"] == "Y" else "0. non-conspiracy"
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

with open("loco_test_conspemollm_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")

with open("loco_test_conspllm_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")


# Process input required for Emotion intensity for EmoLLM - joy
processed_lst = []
for i, row in f1.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nEmotion: joy\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

for i, row in f2.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nEmotion: joy\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

with open("loco_test_emollm_joy_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")

# Process input required for Emotion intensity for EmoLLM - fear
processed_lst = []
for i, row in f1.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nEmotion: fear\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

for i, row in f2.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nEmotion: fear\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

with open("loco_test_emollm_fear_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")


# Process input required for Emotion intensity for EmoLLM - anger
processed_lst = []
for i, row in f1.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nEmotion: anger\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

for i, row in f2.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nEmotion: anger\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

with open("loco_test_emollm_anger_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")


# Process input required for Emotion intensity for EmoLLM - sadness
processed_lst = []
for i, row in f1.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nEmotion: sadness\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

for i, row in f2.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nEmotion: sadness\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

with open("loco_test_emollm_sadness_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")

# Process input required for Sentiment strength for EmoLLM
processed_lst = []
for i, row in f1.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Evaluate the valence intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (most negative) to 1 (most positive).\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

for i, row in f2.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Evaluate the valence intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (most negative) to 1 (most positive).\nText: {}\nIntensity Score:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

with open("loco_test_emollm_sentiment_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")


# Process input required for Sentiment class for EmoLLM
processed_lst = []
for i, row in f1.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Categorize the text into an ordinal class that best characterizes the writer's mental state, considering various degrees of positive and negative sentiment intensity. 3: very positive mental state can be inferred. 2: moderately positive mental state can be inferred. 1: slightly positive mental state can be inferred. 0: neutral or mixed mental state can be inferred. -1: slightly negative mental state can be inferred. -2: moderately negative mental state can be inferred. -3: very negative mental state can be inferred.\nText: {}\nIntensity Class:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

for i, row in f2.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Categorize the text into an ordinal class that best characterizes the writer's mental state, considering various degrees of positive and negative sentiment intensity. 3: very positive mental state can be inferred. 2: moderately positive mental state can be inferred. 1: slightly positive mental state can be inferred. 0: neutral or mixed mental state can be inferred. -1: slightly negative mental state can be inferred. -2: moderately negative mental state can be inferred. -3: very negative mental state can be inferred.\nText: {}\nIntensity Class:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

with open("loco_test_emollm_class_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")


# Process input required for Emotion classification for EmoLLM
processed_lst = []
for i, row in f1.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).\nText: {}\nThis text contains emotions:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

for i, row in f2.iterrows():
    doc_id = row["doc id"]
    loco_text = loco_dataset[doc_id]["txt"]
    loco_text, truncated = truncate_and_check(loco_text)
    prompt = "Task: Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).\nText: {}\nThis text contains emotions:\n".format(
        loco_text
    )
    input = ""
    output = ""
    processed_lst.append(
        {
            "instruction": prompt,
            "input": input,
            "output": output,
            "id": doc_id,
            "truncated": truncated,
        }
    )

with open("loco_test_emollm_emotion_400.json", "w") as file:
    for item in processed_lst:
        file.write(json.dumps(item) + "\n")
