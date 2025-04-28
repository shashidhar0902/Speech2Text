import os
import numpy as np
import time
from sentence_transformers import SentenceTransformer

# Load pretrained model
#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L12-v2')

# Example intent sentences
on_light_examples = [
    "switch on the light",
    "turn on the lamp",
    "light up the room",
    "activate the light",
    "brighten the room",
    "please turn the light on",
    "can you switch on the light",
    "lights on"
]

off_light_examples = [
    "switch off the light",
    "turn off the lamp",
    "darken the room",
    "deactivate the light",
    "dim the lights",
    "please turn the light off",
    "can you switch off the light",
    "lights off"
]

off_fan_examples = [
    "turn off the fan",
    "switch off the fan",
    "fan off",
    "deactivate the fan",
    "dim the fan",
    "please turn the fan off",
    "can you switch off the fan",
    "stop the fan"
]

on_fan_examples = [
    "turn on the fan",
    "switch on the fan",
    "fan on",
    "activate the fan",
    "brighten the fan",
    "please turn the fan on",
    "can you switch on the fan",
    "start the fan"
]

# File paths to save embeddings
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)
on_light_path = os.path.join(embedding_dir, "on_light.npy")
off_light_path = os.path.join(embedding_dir, "off_light.npy")
on_fan_path = os.path.join(embedding_dir, "on_fan.npy")
off_fan_path = os.path.join(embedding_dir, "off_fan.npy")

def save_embeddings():
    np.save(on_light_path, model.encode(on_light_examples))
    np.save(off_light_path, model.encode(off_light_examples))
    np.save(on_fan_path, model.encode(on_fan_examples))
    np.save(off_fan_path, model.encode(off_fan_examples))    

# Save embeddings if not already saved
if not (os.path.exists(on_light_path) and os.path.exists(off_light_path) and
        os.path.exists(on_fan_path) and os.path.exists(off_fan_path)):
    save_embeddings()

# Load embeddings
on_light_embeddings = np.load(on_light_path)
off_light_embeddings = np.load(off_light_path)
on_fan_embeddings = np.load(on_fan_path)
off_fan_embeddings = np.load(off_fan_path)

def detect_intent(prompt):
    """
    Detect intent to switch light or fan on or off based on semantic similarity.
    Returns one of 'on_light', 'off_light', 'on_fan', 'off_fan', or 'unknown'.
    """
    # start_time = time.time()
    prompt_embedding = model.encode([prompt])[0]
    # end_time = time.time()
    # exec_time = end_time - start_time
    # with open("execution_time.txt", "w") as f:
    #     f.write(f"Execution time: {exec_time} seconds")
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = {
        "on_light": max(cosine_similarity(prompt_embedding, emb) for emb in on_light_embeddings),
        "off_light": max(cosine_similarity(prompt_embedding, emb) for emb in off_light_embeddings),
        "on_fan": max(cosine_similarity(prompt_embedding, emb) for emb in on_fan_embeddings),
        "off_fan": max(cosine_similarity(prompt_embedding, emb) for emb in off_fan_embeddings)
    }

    threshold = 0.4  # lowered threshold to be more inclusive
    best_intent = max(similarities, key=similarities.get)
    if similarities[best_intent] > threshold:
        return best_intent
    else:
        return "unknown"
