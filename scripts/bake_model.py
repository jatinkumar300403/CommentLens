"""Save a single local copy of a Hugging Face sentiment model, for baking into the Docker image.

The Hub repo for the default model ships its weights twice (pytorch_model.bin and
model.safetensors), so loading it straight from the Hub cache puts 2.2 GB in the image.
Re-saving writes one safetensors copy; the Dockerfile deletes the Hub cache in the same layer.

Usage: python bake_model.py HUB_MODEL_ID TARGET_DIR
"""
import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer

source, target = sys.argv[1], sys.argv[2]
AutoTokenizer.from_pretrained(source).save_pretrained(target)
AutoModelForSequenceClassification.from_pretrained(source).save_pretrained(target)
print(f'saved {source} to {target}')
