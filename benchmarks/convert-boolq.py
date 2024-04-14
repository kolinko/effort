# for converting the parquet file with boolq
# https://huggingface.co/datasets/google/boolq/tree/main/data

import parquet
import json

#    "https://storage.googleapis.com/boolq/dev.jsonl"

data = []

with open("data/dev.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

with open("data/dev.json", "w") as f:
    json.dump(data, f, indent=2)

