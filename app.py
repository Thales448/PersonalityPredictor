import os
import json
import torch
from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
from flask_cors import CORS
import traceback
from datetime import datetime

# Prevent HF from attempting to pull from the internet
os.environ['TRANSFORMERS_OFFLINE'] = '1'

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.155:3000",
    "http://192.168.1.126:3000",
    "http://localhost",
    "http://192.168.1.155",
    "https://litproject.rodrialexander.com",
    "https://api.litproject.rodrialexander.com"
]}})

# Model setup
model_path = "/home/psych/workspace/models/bertbasePersonality"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5)
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model.eval()

model.config.label2id = {
    "Extroversion": 0,
    "Neuroticism": 1,
    "Agreeableness": 2,
    "Conscientiousness": 3,
    "Openness": 4,
}
model.config.id2label = {str(v): k for k, v in model.config.label2id.items()}

def personality_detection(text: str) -> dict:
    if len(text.strip()) == 0:
        return {label: 0.0 for label in model.config.label2id}

    encoded = tokenizer.encode_plus(
        text, max_length=512, padding='max_length', truncation=True, return_tensors="pt"
    )
    input_ids = torch.stack([encoded['input_ids'][0]] * 2)
    attention_mask = torch.stack([encoded['attention_mask'][0]] * 2)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        scores = torch.sigmoid(logits)[0]

    return {
        "Extroversion": float(scores[0]),
        "Neuroticism": float(scores[1]),
        "Agreeableness": float(scores[2]),
        "Conscientiousness": float(scores[3]),
        "Openness": float(scores[4]),
    }

traits_file = "/home/psych/workspace/PersonalityPredictor/character_embeddings_bigfive.json"
description_file = "/home/psych/workspace/PersonalityPredictor/descriptions.json"
character_traits = {}
character_descriptions = {}

if os.path.exists(traits_file):
    print("\U0001f9e0 Loading precomputed character traits...")
    with open(traits_file) as f:
        character_traits = json.load(f)
else:
    print("\U0001f9e0 Computing character traits from wikis...")
    with open("/home/psych/workspace/PersonalityPredictor/character_wikis.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    entries = raw.split("### ")
    for entry in entries:
        if not entry.strip():
            continue
        try:
            name, content = entry.split("###\n", 1)
            name, content = name.strip(), content.strip()
        except ValueError:
            continue
        print(f"Scoring: {name}")
        character_traits[name] = personality_detection(content)

    with open(traits_file, "w") as f:
        json.dump(character_traits, f, indent=2)
    print(f"\u2705 Saved character traits to {traits_file}")

if os.path.exists(description_file):
    with open(description_file, "r") as f:
        descriptions = json.load(f)
        character_descriptions = {
            entry["name"]: entry["description"]
            for entry in descriptions.get("characters", [])
        }

def cosine_similarity(a: dict, b: dict) -> float:
    vec_a = torch.tensor([a[k] for k in model.config.label2id])
    vec_b = torch.tensor([b[k] for k in model.config.label2id])
    return torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0).item()

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok"})

@app.route("/score", methods=["POST"])
def score():
    print("Request headers:", dict(request.headers))
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid format — must be an object with q# keys."}), 400

    question_map = {
        "q1": "You are a ruler facing a rebellious people. Do you offer them amnesty, crush them brutally, or something else?",
        "q2": "You are in exile and someone offers you the chance to return by betraying your ideals. What do you do?",
        "q3": "You are granted the power to speak to a crowd once. What message do you share?",
        "q4": "A rival challenges your legacy in front of an audience. How do you respond?",
        "q5": "You have written a controversial poem that could lead to exile or fame. Do you publish it?",
        "q6": "You discover a prophecy that foretells your downfall. Do you try to change it or accept it?",
        "q7": "If you could be remembered for one act alone, what would it be?"
    }

    formatted = [f"Q: {question_map[k]} A: {v.strip()}" for k, v in data.items() if k in question_map and v.strip()]

    if len(formatted) < 1:
        return jsonify({"error": "No valid question/answer pairs provided."}), 400

    # Save user answers
    try:
        log_path = os.path.join(os.path.dirname(traits_file), "user_answers_log.jsonl")
        with open(log_path, "a") as log_file:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "answers": {k: v.strip() for k, v in data.items() if k in question_map}
            }
            log_file.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print("\u26a0\ufe0f Could not save user answers:", e)

    traits_list = [personality_detection(text) for text in formatted]
    averaged_traits = {
        trait: sum(t[trait] for t in traits_list) / len(traits_list)
        for trait in model.config.label2id
    }

    best_match = None
    best_score = -1
    best_traits = {}
    for name, traits in character_traits.items():
        score = cosine_similarity(averaged_traits, traits)
        if score > best_score:
            best_score = score
            best_match = name
            best_traits = traits

    description = character_descriptions.get(best_match, "")

    return jsonify({
        "user_traits": averaged_traits,
        "closest_match": best_match,
        "closest_match_traits": best_traits,
        "description": description
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051)
