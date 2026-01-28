import json
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# load Teachable Machine image model
TM_MODEL_PATH = "keras_model.h5"
TM_LABELS_PATH = "labels.txt"

image_model = tf.keras.models.load_model(TM_MODEL_PATH, compile=False)

# load class labels
with open(TM_LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip() != ""]


def predict_image_label(image_path):
    # given a path to an image file or webcam, return:
    # - predicted class (item) name
    # - confidence score (0–1)

    # Teachable Machine usually uses 224x224 images and [-1,1] normalisation
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.asarray(img).astype(np.float32)
    # scale to [-1, 1]
    normalised = (img_array / 127.5) - 1.0

    # shape (1, 224, 224, 3) 1 img, height, width, 3 colour channels
    data = np.expand_dims(normalised, axis=0)

    predictions = image_model.predict(data, verbose=0)[0]
    idx = int(np.argmax(predictions))
    confidence = float(predictions[idx])

    class_name = class_names[idx]

    return class_name, confidence


def predict_image_from_frame(frame):
    # take a webcam frame preprocess it like the Teachable Machine image and return (class_name, confidence)

    # convert from BGR (OpenCV) to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img_array = img.astype(np.float32)
    normalised = (img_array / 127.5) - 1.0

    data = np.expand_dims(normalised, axis=0)

    predictions = image_model.predict(data, verbose=0)[0]
    idx = int(np.argmax(predictions))
    confidence = float(predictions[idx])
    class_name = class_names[idx]

    return class_name, confidence


# load task 1 text model (for grocery category)

TASK1_MODEL_PATH = "../task-1/task1_model_task3_env.keras"
TASK1_LABEL_MAP_PATH = "../task-1/task1_label_map_task3_env.json"

text_model = tf.keras.models.load_model(TASK1_MODEL_PATH)

with open(TASK1_LABEL_MAP_PATH, "r") as f:
    label_data = json.load(f)

# JSON keys are strings, convert back to ints
idx_to_cat = {}
for k, v in label_data.items():
    idx_to_cat[int(k)] = v


def clean_text(text):
    text = str(text).lower().strip()
    text = " ".join(text.split())
    return text


def predict_text_category(item_name):
    # task 1 model to classify an item name into a grocery category
    item = clean_text(item_name)
    x = np.array([item], dtype=object)

    probs = text_model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    category = idx_to_cat[idx]
    confidence = float(probs[idx])

    return category, confidence



def camera_mode():
    # press 'c' to capture & classify the current frame
    # press 'q' to quit

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Camera mode:")
    print("  - Press 'c' to capture and classify")
    print("  - Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # show the live video
        cv2.imshow("Task 3 Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # classify this frame
            product_name, image_conf = predict_image_from_frame(frame)

            grocery_category, text_conf = predict_text_category(product_name)

            print("\nCaptured frame classified as:")
            print(f"  Product: {product_name} (image confidence: {image_conf:.2f})")
            print(f"  Category: {grocery_category} (text confidence: {text_conf:.2f})")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# command-line interface

def main():
    print("Task 3: Image-based grocery recogniser")
    print("Choose mode:")
    print("  1 - Image path mode")
    print("  2 - Camera mode\n")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        camera_mode()
        return

    print("\nImage path mode (type 'q' to quit).\n")

    while True:
        path = input("Enter image path: ").strip()
        if path.lower() in ["q", "quit", "exit"]:
            break

        try:
            product_name, image_conf = predict_image_label(path)
            grocery_category, text_conf = predict_text_category(product_name)

            print(f"Product: {product_name} ({image_conf:.2f})")
            print(f"Category: {grocery_category} ({text_conf:.2f})\n")
            print()
        except FileNotFoundError:
            print("Could not open that image file. Please check the path.\n")
        except Exception as e:
            print("Something went wrong:", e, "\n")

if __name__ == "__main__":
    main()