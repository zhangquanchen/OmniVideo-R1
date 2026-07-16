#!/usr/bin/env python3
"""
Dataset Split Script
Split training and test sets from JSONL file by category
"""

import json
import random
from collections import defaultdict
from pathlib import Path
import argparse

# Set random seed for reproducibility
random.seed(42)

# Default configuration parameters
DEFAULT_INPUT_FILE = "./data/input.jsonl"
DEFAULT_OUTPUT_DIR = "./data"
DEFAULT_TEST_SAMPLES_PER_CATEGORY = 2000

def main():
    parser = argparse.ArgumentParser(description="Dataset split script")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_FILE, help="Input JSONL file path")
    parser.add_argument("--output_dir", "-o", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--test_samples", "-t", type=int, default=DEFAULT_TEST_SAMPLES_PER_CATEGORY, 
                       help="Number of test samples per category (used when category count > 20000)")
    args = parser.parse_args()
    
    INPUT_FILE = args.input
    OUTPUT_DIR = args.output_dir
    TEST_SAMPLES_PER_CATEGORY = args.test_samples
    
    print("=" * 60)
    print("Dataset Split Script")
    print("=" * 60)
    
    # Read data and group by category
    print(f"\nReading data file: {INPUT_FILE}")
    data_by_type = defaultdict(list)
    total_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                category = data.get('Type', 'unknown')
                data_by_type[category].append(data)
                total_count += 1
    
    print(f"Total {total_count} entries read")
    print(f"Total {len(data_by_type)} categories\n")
    
    # Display data distribution for each category
    print("Category data distribution:")
    print("-" * 60)
    for category in sorted(data_by_type.keys()):
        count = len(data_by_type[category])
        print(f"  {category}: {count} entries")
    print("-" * 60)
    
    # Split training and test sets
    train_data = []
    test_data = []
    
    print(f"\nStarting dataset split:\n")
    print("Sampling strategy:")
    print(f"  - Category count > 20000: test set samples {TEST_SAMPLES_PER_CATEGORY} entries")
    print(f"  - Category count <= 20000: test set samples 1/10 of category count\n")
    
    for category, items in sorted(data_by_type.items()):
        # Randomly shuffle data
        random.shuffle(items)
        
        # Determine test set size
        if len(items) > 20000:
            # More than 20000, sample 2000
            test_size = TEST_SAMPLES_PER_CATEGORY
        else:
            # 20000 or less, sample 1/10
            test_size = max(1, int(len(items) * 0.1))  # Ensure at least 1
        
        # Split
        test_samples = items[:test_size]
        train_samples = items[test_size:]
        
        test_data.extend(test_samples)
        train_data.extend(train_samples)
        
        print(f"  {category}:")
        print(f"    - Test set: {len(test_samples)} entries")
        print(f"    - Training set: {len(train_samples)} entries")
    
    # Output statistics
    print("\n" + "=" * 60)
    print(f"Split results:")
    print(f"  Training set total: {len(train_data)} entries")
    print(f"  Test set total: {len(test_data)} entries")
    print(f"  Total: {len(train_data) + len(test_data)} entries")
    print("=" * 60)
    
    # Save training set
    train_output = Path(OUTPUT_DIR) / "train_set.jsonl"
    print(f"\nSaving training set to: {train_output}")
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Training set saved successfully!")
    
    # Save test set
    test_output = Path(OUTPUT_DIR) / "test_set.jsonl"
    print(f"Saving test set to: {test_output}")
    with open(test_output, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Test set saved successfully!")
    
    print("\n✓ Dataset split complete!")
    print(f"  - Training set: {train_output}")
    print(f"  - Test set: {test_output}")

if __name__ == "__main__":
    main()
