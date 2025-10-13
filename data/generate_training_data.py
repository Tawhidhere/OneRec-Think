#!/usr/bin/env python3
import json
import pandas as pd

def generate_training_data(sequential_file, beauty_items_file, output_train_file, output_val_file, output_test_file):
    try:
        with open(beauty_items_file, 'r', encoding='utf-8') as f:
            beauty_items = json.load(f)

        with open(sequential_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        training_data_train = []
        training_data_val = []
        training_data_test = []
        missing_items_count = 0

        def build_description(item_ids, user_id):
            item_descriptions = []
            local_missing_count = 0

            for item_id in item_ids:
                item_info = beauty_items.get(item_id)
                if not item_info:
                    local_missing_count += 1
                    continue

                sid = item_info.get('sid', '')
                title = item_info.get('title', '')
                categories = item_info.get('categories', '')

                if sid and title and categories:
                    item_desc = f'{sid}, its title is "{title}", its categories are "{categories}"'
                    item_descriptions.append(item_desc)
                else:
                    local_missing_count += 1

            return item_descriptions, local_missing_count

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            elements = line.split()
            if len(elements) <= 1:
                continue

            user_id = elements[0]
            item_ids = elements[1:]

            full_item_descriptions, missing_count_full = build_description(item_ids, user_id)
            missing_items_count += missing_count_full

            if full_item_descriptions:
                test_description = "The user has purchased the following items: " + "; ".join(full_item_descriptions) + ";"
                training_data_test.append({
                    'user_id': user_id,
                    'description': test_description
                })

            if len(item_ids) > 1:
                val_item_ids = item_ids[:-1]
                val_item_descriptions, missing_count_val = build_description(val_item_ids, user_id)
                missing_items_count += missing_count_val

                if val_item_descriptions:
                    val_description = "The user has purchased the following items: " + "; ".join(val_item_descriptions) + ";"
                    training_data_val.append({
                        'user_id': user_id,
                        'description': val_description
                    })

            if len(item_ids) > 2:
                train_item_ids = item_ids[:-2]
                train_item_descriptions, missing_count_train = build_description(train_item_ids, user_id)
                missing_items_count += missing_count_train

                if train_item_descriptions:
                    train_description = "The user has purchased the following items: " + "; ".join(train_item_descriptions) + ";"
                    training_data_train.append({
                        'user_id': user_id,
                        'description': train_description
                    })

        df_train = pd.DataFrame(training_data_train)
        df_val = pd.DataFrame(training_data_val)
        df_test = pd.DataFrame(training_data_test)

        df_train.to_parquet(output_train_file, engine='pyarrow', index=False)
        df_val.to_parquet(output_val_file, engine='pyarrow', index=False)
        df_test.to_parquet(output_test_file, engine='pyarrow', index=False)

    except Exception as e:
        print(f"Failed to generate training data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sequential_file = "./sequential_data_processed.txt"
    beauty_items_file = "./Beauty.pretrain.json"
    output_train_file = "./training_align_data_train.parquet"
    output_val_file = "./training_align_data_val.parquet"
    output_test_file = "./training_align_data_test.parquet"

    generate_training_data(
        sequential_file,
        beauty_items_file,
        output_train_file,
        output_val_file,
        output_test_file
    )
