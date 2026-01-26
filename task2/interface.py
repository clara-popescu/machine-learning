# interface.py
# Command-line interface for Task 2 grocery recommendation

import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf


# load task 1 text model
item_model = tf.keras.models.load_model("../task-1/model.keras")

with open("../task-1/label_map.json", "r") as f:
    label_data = json.load(f)

# task 1 - map index to category name
idx_to_label = {}
for k, v in label_data.items():
    idx_to_label[int(k)] = v


def clean_text(text):
    text = str(text).lower().strip()
    text = " ".join(text.split())
    return text


def predict_category(item_name):
    # use task 1 model to turn an item name into a category
    item = clean_text(item_name)
    # task 1 model expects array of strings
    x = np.array([item], dtype=object)  

    probs = item_model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx_to_label[idx]


def get_categories_for_basket(item_list):
    # Get a sorted list of unique categories for a basket of item names.
    cats = []

    for item in item_list:
        cat = predict_category(item)
        cats.append(cat)

    cats = sorted(list(set(cats)))
    return cats


# load task 2 model and category list
rec_model = tf.keras.models.load_model("rec_model.keras")

with open("all_categories.json", "r") as f:
    all_categories = json.load(f)

all_categories = list(all_categories)

# reconstructing category to index mapping to make sure that the output corresponds to the right label
cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
idx_to_cat = {i: cat for i, cat in enumerate(all_categories)}
num_categories = len(all_categories)


def make_multi_hot(categories):
    # turn list of category names into a multi-hot vector
    vec = np.zeros(num_categories, dtype=np.float32)

    for c in categories:
        if c in cat_to_idx:
            vec[cat_to_idx[c]] = 1.0

    return vec


# load items dataset + mapping

all_items = pd.read_csv("../task-1/data/final_grocery_dataset.csv")

category_mapping = {
    "Grains & Bakery": ["Bakery", "Pasta & Grains"],
    "Pantry Items": ["Pantry", "Canned Goods", "Condiments & Sauces"],
    "Meat & Deli": ["Meat & Seafood", "Deli"],
    "Beverages": ["Beverages"],
    "Dairy & Eggs": ["Dairy & Eggs"],
    "Frozen Foods": ["Frozen Foods"],
    "Produce": ["Produce"],
    "Snacks": ["Snacks"],
    "Household": ["Household"],
    "Personal Care": ["Personal Care"],
    "Pet Supplies": ["Pet Supplies"],
    "Other": ["Other"],
}


def get_random_item_from_category(model_category):
    if model_category not in category_mapping:
        return None

    valid_categories = category_mapping[model_category]

    possible_items = all_items[all_items["Category"].isin(valid_categories)]["Item"].tolist()

    if len(possible_items) == 0:
        return None

    return random.choice(possible_items)


# main recommendation logic

# full pipeline:
# 1) item names -> Task 1 categories
# 2) categories -> multi-hot vector
# 3) Task 2 -> extra category
# 4) extra category -> random item


def recommend_extra_item(item_list):
    cats = get_categories_for_basket(item_list)

    if len(cats) == 0:
        return None, None

    x_vec = make_multi_hot(cats).reshape(1, -1)

    probs = rec_model.predict(x_vec, verbose=0)[0]

    # do not recommend categories already in basket
    for c in cats:
        if c in cat_to_idx:
            probs[cat_to_idx[c]] = 0.0

    best_idx = int(np.argmax(probs))
    best_category = idx_to_cat[best_idx]

    item = get_random_item_from_category(best_category)

    return item, best_category


# command line interface

def main():
    print("Grocery Basket Recommender (Task 2)")
    print("Type 'q' at any prompt to quit.\n")

    while True:
        item1 = input("Enter first item: ").strip()
        if item1.lower() in ["q", "quit", "exit"]:
            break

        item2 = input("Enter second item: ").strip()
        if item2.lower() in ["q", "quit", "exit"]:
            break

        basket = [item1, item2]

        rec_item, rec_cat = recommend_extra_item(basket)

        print("\nYour basket:", basket)

        if rec_item is None:
            print("→ Sorry, I couldn't find a recommendation.\n")
        else:
            print(f"→ Recommended category: {rec_cat}")
            print(f"→ Recommended item: {rec_item}\n")


if __name__ == "__main__":
    main()
