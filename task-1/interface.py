import json
import tensorflow as tf

# 1) Load saved model
model = tf.keras.models.load_model("model.keras")

# 2) Load labels (saved during training)
with open("label_map.json", "r") as f:
    idx_to_cat = {int(k): v for k, v in json.load(f).items()}

def clean_text(s: str) -> str:
    return " ".join(str(s).lower().strip().split())

def predict_category(item_name: str, threshold=0.5):
    item = clean_text(item_name)
    x = tf.constant([[item]])
    probs = model.predict(x, verbose=0)[0]
    max_prob = float(probs.max())
    pred_idx = int(probs.argmax())

    if max_prob < threshold:
        return "Other", max_prob

    return idx_to_cat[pred_idx], max_prob

if __name__ == "__main__":
    print("Grocery categoriser (type 'q' to quit)\n")
    while True:
        item = input("Enter item: ").strip()
        if item.lower() in {"q", "quit", "exit"}:
            break

        category, conf = predict_category(item)
        print(f"→ {category} (confidence: {conf:.2f})\n")
