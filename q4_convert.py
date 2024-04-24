'''
 unlikely to run out of the box - you probably will need to fix the dirs at least.
'''


import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import pdb
import json
from q4_draft import convert
import torch

# hardcoded dir, sorry about that. this was tested and works with hf @ mistralai/Mistral-7B-Instruct-v0.2 
tensors = {}
print("loading source model")

for i in range(3):
    with safe_open(f"../models/model-0000{i+1}-of-00003.safetensors", framework="pt") as f:
        for k in f.keys():
            tmp_tensor = f.get_tensor(k)
            tensor_float32 = tmp_tensor.to(dtype=torch.float32).cpu()
            tensor_float16 = tensor_float32.to(dtype=torch.float16)
            tensors[k] = tensor_float16.numpy()

# btw. damn you mistralai for changing the namings mid-release of a model!

out_tensors = []
out_tensors.append({
    "model.norm": tensors["model.norm.weight"],
    "output.core": tensors["lm_head.weight"],
    "tok_embeddings.core": tensors["model.embed_tokens.weight"]
})


#oldPrefix = f"model.layers.0.mlp.gate_proj"
#buckets = convert(tensors[f"{oldPrefix}.weight"].T)


numLayers = 32
for i in range(numLayers):
    print(f"converting layer {i}")

    out = {}
    out[f"layers.{i}.attention_norm"] = tensors[f"model.layers.{i}.input_layernorm.weight"]
    out[f"layers.{i}.ffn_norm"] = tensors[f"model.layers.{i}.post_attention_layernorm.weight"]

    for s in ["k", "o", "q", "v"]:
        print(s)
        oldPrefix = f"model.layers.{i}.self_attn.{s}_proj.weight"
        newPrefix = f"layers.{i}.attention.w{s}."
        out[newPrefix + "core"] = tensors[oldPrefix]
        if s not in ["k", "o", "v"]:
            buckets = convert(tensors[oldPrefix].T)

            for k, t in buckets.items():
                out[newPrefix + k] = t

    for oldName, newName in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")]:
        oldPrefix = f"model.layers.{i}.mlp."
        newPrefix = f"layers.{i}.feed_forward.experts.0."

        out[f"{newPrefix}{newName}.core"] = tensors[f"{oldPrefix}{oldName}.weight"]
        buckets = convert(tensors[f"{oldPrefix}{oldName}.weight"].T)
        for k, t in buckets.items():
            out[f"{newPrefix}{newName}.{k}"] = t

    out_tensors.append(out)

wm = {}
for i in range(len(out_tensors)):
    fname = f"model-{i+1:05d}-of-{len(out_tensors)}.safetensors"
    for k in out_tensors[i]:
        wm[k] = fname

    save_file(out_tensors[i], "../models/mistral-q4/"+fname)

index = {"weight_map": wm}

with open("../models/mistral-q4/model.safetensors.index.json", "w") as f:
    f.write(json.dumps(index, indent=2))

print("done.")