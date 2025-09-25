#!/usr/bin/env python3

import pandas as pd
from load_ucirepo import get_ucidata
datasets = [
    ('iris', 53, 'classification'),            # 150
    ('wine', 109, 'classification'),           # 178
    ('hearth', 45, 'classification'),          # 303
    ('realstate', 477, 'regression'),          # 414
    ('breast', 17, 'classification'),          # 569
    ('student_perf', 320, 'regression'),       # 649
    ('energy_efficiency', 242, 'regression'),  # 768
    ('concrete', 165, 'regression'),           # 1030
    ('car_evaluation', 19, 'classification'),  # 1728
    ('obesity', 544, 'regression'),            # 2111
    ('abalone', 1, 'regression'),              # 4177
    ('student_dropout', 697, 'classification'),# 4424
    ('winequalityc', 186, 'classification'),   # 6497
    ('mushrooms', 73, 'classification'),       # 8124
    ('ai4i', 601, 'regression'),               # 10000
    ('bike', 275, 'regression'),               # 17379
    ('appliances', 374, 'regression'),         # 19735
    ('popularity', 332, 'regression'),         # 39644
    ('bank', 222, 'classification'),           # 45211
    ('adult', 2, 'classification'),            # 48842
    ('seoulBike', 560, 'regression'),
]

def create_dataset_table(datasets_list, cap=50):
    """
    Create a table with dataset statistics using get_ucidata function.

    Args:
        datasets_list: List of tuples (name, id, task_type)
        cap: Feature capacity limit for one-hot encoding

    Returns:
        pandas.DataFrame with dataset statistics
    """
    results = []

    for name, dataset_id, task in datasets_list:
        try:
            print(f"Processing {name} (ID: {dataset_id})...")

            # Load and preprocess the dataset
            X_train, y_train, X_val, y_val, X_test, y_test = get_ucidata(
                dataset_id, task, device='cpu', cap=cap
            )

            # Extract statistics
            row = {
                'ID': dataset_id,
                'Dataset': name,
                'Task': task,
                'Train': X_train.shape[0],
                'Val': X_val.shape[0],
                'Test': X_test.shape[0],
                'Features': X_train.shape[1]
            }
            results.append(row)

        except Exception as e:
            print(f"Error processing {name} (ID: {dataset_id}): {e}")
            continue

    return pd.DataFrame(results)

def print_table(df):
    """Print DataFrame as a markdown table."""
    print("| ID  | Dataset            | Task           | Train | Val  | Test | Features |")
    print("|-----|--------------------|----------------|-------|------|------|----------|")

    for _, row in df.iterrows():
        print(f"| {row['ID']:<3} | {row['Dataset']:<18} | {row['Task']:<14} | {row['Train']:<5} | {row['Val']:<4} | {row['Test']:<4} | {row['Features']:<8} |")

if __name__ == "__main__":
    # Create the table
    df = create_dataset_table(datasets)

    # Print as markdown table
    print_table(df)

    # Save to CSV
    df.to_csv('dataset_statistics.csv', index=False)
    print(f"\nTable saved to dataset_statistics.csv")