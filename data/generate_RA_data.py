#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_beauty_items(beauty_items_file: Path) -> dict:
    print(f"Loading Beauty items file: {beauty_items_file}")
    with beauty_items_file.open("r", encoding="utf-8") as f:
        beauty_items = json.load(f)
    print(f"Beauty items count: {len(beauty_items)}")
    return beauty_items


def extract_sid_sequence(item_ids: list[str], user_id: str, beauty_items: dict) -> list[dict]:
    item_sequence: list[dict] = []
    for item_id in item_ids:
        item_info = beauty_items.get(item_id)
        if not item_info:
            print(f"Warning: item_id {item_id} for user {user_id} not found in Beauty items, skipping.")
            continue
        sid = item_info.get("sid")
        if not sid:
            print(f"Warning: item_id {item_id} for user {user_id} missing sid field, skipping.")
            continue

        title = item_info.get("title", "")
        categories = item_info.get("categories", "")

        item_sequence.append({
            "sid": sid,
            "title": title,
            "categories": categories
        })
    return item_sequence


def build_dataset_entry(
    user_id: str,
    item_sequence: list[dict],
    tail_remove_count: int,
) -> dict | None:
    if len(item_sequence) <= tail_remove_count + 1:
        return None

    candidate_sequence = (
        item_sequence[: len(item_sequence) - tail_remove_count]
        if tail_remove_count > 0
        else item_sequence
    )

    if len(candidate_sequence) < 2:
        return None

    groundtruth_item = candidate_sequence[-1]
    description_items = candidate_sequence[:-1]

    if not description_items:
        return None

    item_descriptions = []
    for item in description_items:
        sid = item["sid"]
        title = item["title"]
        categories = item["categories"]

        item_desc = f'{sid}, its title is "{title}", its categories are "{categories}"'
        item_descriptions.append(item_desc)

    description = "The user has purchased the following items: " + "; ".join(item_descriptions) + ";"

    return {
        "user_id": user_id,
        "description": description,
        "groundtruth": groundtruth_item["sid"],
        "title": groundtruth_item["title"],
        "categories": groundtruth_item["categories"],
    }


def generate_sid_prediction_data(
    sequential_file: Path,
    beauty_items_file: Path,
    output_train: Path,
    output_val: Path,
    output_test: Path,
) -> None:
    beauty_items = load_beauty_items(beauty_items_file)

    print(f"Loading Sequential data file: {sequential_file}")
    with sequential_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Sequential data lines: {len(lines)}")

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        elements = line.split()
        if len(elements) <= 1:
            continue

        user_id = elements[0]
        item_ids = elements[1:]

        item_sequence = extract_sid_sequence(item_ids, user_id, beauty_items)
        if len(item_sequence) == 0:
            continue

        entry_train = build_dataset_entry(user_id, item_sequence, tail_remove_count=2)
        if entry_train:
            train_rows.append(entry_train)

        entry_val = build_dataset_entry(user_id, item_sequence, tail_remove_count=1)
        if entry_val:
            val_rows.append(entry_val)

        entry_test = build_dataset_entry(user_id, item_sequence, tail_remove_count=0)
        if entry_test:
            test_rows.append(entry_test)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} lines...")

    print("Creating DataFrame...")
    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    print(f"Training set entries: {len(df_train)}")
    print(f"Validation set entries: {len(df_val)}")
    print(f"Test set entries: {len(df_test)}")

    print(f"Saving training set to: {output_train}")
    df_train.to_parquet(output_train, engine="pyarrow", index=False)

    print(f"Saving validation set to: {output_val}")
    df_val.to_parquet(output_val, engine="pyarrow", index=False)

    print(f"Saving test set to: {output_test}")
    df_test.to_parquet(output_test, engine="pyarrow", index=False)

    def preview(df: pd.DataFrame, name: str) -> None:
        print(f"\n{name} first 2 rows preview:")
        for _, row in df.head(2).iterrows():
            print(f"user_id {row['user_id']}")
            print(f"description: {row['description']}")
            print(f"groundtruth: {row['groundtruth']}")
            print(f"title: {row['title']}")
            print(f"categories: {row['categories']}")

    preview(df_train, "Training set")
    preview(df_val, "Validation set")
    preview(df_test, "Test set")


if __name__ == "__main__":
    sequential_file = Path("./sequential_data_processed.txt")
    beauty_items_file = Path("./Beauty.pretrain.json")
    output_train = Path("./training_RA_train.parquet")
    output_val = Path("./training_RA_val.parquet")
    output_test = Path("./training_RA_test.parquet")

    generate_sid_prediction_data(
        sequential_file=sequential_file,
        beauty_items_file=beauty_items_file,
        output_train=output_train,
        output_val=output_val,
        output_test=output_test,
    )
