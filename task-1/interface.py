import json
import tensorflow as tf

# 1) load saved model
model = tf.keras.models.load_model("model.keras")

# 2) load labels
with open("label_map.json", "r") as f:
    label_data = json.load(f)

# JSON keys are strings, convert them back to ints
idx_to_cat = {}
for k, v in label_data.items():
    idx_to_cat[int(k)] = v

confidence_treshold = 0.3

def clean_text(text):
    text = str(text).lower().strip()
    text = " ".join(text.split())
    return text

def predict_category(item_name):
    # clean input
    item = clean_text(item_name)
    x = tf.constant([[item]])

    probs = model.predict(x, verbose=0)[0]
    max_prob = float(probs.max())
    pred_idx = int(probs.argmax())
    pred_cat = idx_to_cat[pred_idx]

    return pred_cat, max_prob

def main():
    print("Grocery categoriser (type 'q' to quit)\n")

    while True:
        item = input("Enter item: ").strip()
        if item.lower() in ["q", "quit", "exit"]:
            break

        category, conf = predict_category(item)

        if conf < confidence_treshold:
            print(
                f"→ I'm not confident enough to classify this item "
                f"(best guess: {category}, {conf:.2f})\n"
            )
        else:
            print(f"→ {category} (confidence: {conf:.2f})\n")

if __name__ == "__main__":
    main()
