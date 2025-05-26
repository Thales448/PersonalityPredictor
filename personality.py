import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib
import os
# Load model directly

# Use Agg backend for headless environments
matplotlib.use('Agg')

# Load precomputed character descriptions
with open("/home/psych/workspace/PersonalityPredictor/character_wikis.txt", "r", encoding="utf-8") as f:
    raw = f.read()

entries = raw.split("### ")
characters = {}
for entry in entries:
    if not entry.strip():
        continue
    try:
        name, content = entry.split("###\n", 1)
        characters[name.strip()] = content.strip()
    except ValueError:
        continue

# Load Big Five personality classifier model
model = BertForSequenceClassification.from_pretrained("/home/psych/workspace/models/bertbasePersonality", num_labels=5)
tokenizer = BertTokenizer.from_pretrained('/home/psych/workspace/models/bertbasePersonality', do_lower_case=True)
model.eval()
model.config.label2id = {
    "Extroversion": 0,
    "Neuroticism": 1,
    "Agreeableness": 2,
    "Conscientiousness": 3,
    "Openness": 4,
}
model.config.id2label = {str(v): k for k, v in model.config.label2id.items()}

def personality_detection(model_input: str) -> dict:
    if len(model_input.strip()) == 0:
        return {label: 0.0 for label in model.config.label2id}

    dict_custom = {}
    encoded = tokenizer.encode_plus(model_input, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    dict_custom['input_ids'] = torch.stack([encoded['input_ids'][0], encoded['input_ids'][0]])
    dict_custom['attention_mask'] = torch.stack([encoded['attention_mask'][0], encoded['attention_mask'][0]])

    with torch.no_grad():
        outs = model(input_ids=dict_custom['input_ids'], attention_mask=dict_custom['attention_mask'])
    logits = outs.logits
    pred_scores = torch.sigmoid(logits)

    return {
        "Extroversion": float(pred_scores[0][0]),
        "Neuroticism": float(pred_scores[0][1]),
        "Agreeableness": float(pred_scores[0][2]),
        "Conscientiousness": float(pred_scores[0][3]),
        "Openness": float(pred_scores[0][4]),
    }

# Generate Big Five vectors for all characters
character_traits = {}
for name, text in characters.items():
    print(f"Scoring: {name}")
    scores = personality_detection(text)
    character_traits[name] = scores

# Save results
with open("character_embeddings_bigfive.json", "w") as f:
    json.dump(character_traits, f, indent=2)

# Visualize heatmap
names = list(character_traits.keys())
traits = ["Openness", "Conscientiousness", "Extroversion", "Agreeableness", "Neuroticism"]
data = np.array([[character_traits[name][trait] for trait in traits] for name in names])

plt.figure(figsize=(14, 10))
sns.heatmap(data, xticklabels=traits, yticklabels=names, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Character Big Five Personality Scores", fontsize=16)
plt.xlabel("Trait", fontsize=12)
plt.ylabel("Character", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bigfive_trait_heatmap.png")
