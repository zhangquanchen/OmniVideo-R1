#!/usr/bin/env python3
"""
Data Filtering and Category Balancing Script

Features:
1. Count category_name distribution
2. Remove all data from categories with fewer than 10 instances
3. Remove all data with category 'none'
4. For the largest category A1, if it exceeds 3 times the second largest category A2:
   - First keep data with video_dependency_score=10 and audio_dependency_score=10
   - Then sort the rest by composite_score
   - Keep a total of 3 times A2 data

Usage:
    python filter_and_balance_categories.py --input input.jsonl --output output.jsonl
"""

import json
import argparse
from collections import defaultdict


def get_category_name(data):
    """Get category_name field"""
    question_cls = data.get('question_classification', {})
    return question_cls.get('category_name', 'unknown')


def get_scores(data):
    """Get score information"""
    quality_eval = data.get('quality_evaluation', {})
    scores = quality_eval.get('scores', {})
    return {
        'video_dependency_score': scores.get('video_dependency_score', 0),
        'audio_dependency_score': scores.get('audio_dependency_score', 0),
        'composite_score': scores.get('composite_score', 0)
    }


def main():
    parser = argparse.ArgumentParser(description="Data filtering and category balancing")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    args = parser.parse_args()
    
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    
    # Step 1: Read all data and group by category
    print(f"Reading file: {INPUT_FILE}")
    print("=" * 70)
    
    category_data = defaultdict(list)
    total = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total += 1
            data = json.loads(line)
            category = get_category_name(data)
            category_data[category].append(data)
    
    print(f"Original data count: {total}")
    print(f"Original category count: {len(category_data)}")
    
    # Display original category distribution
    print("\n" + "-" * 70)
    print("Original category distribution:")
    print("-" * 70)
    for cat, items in sorted(category_data.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat}: {len(items)}")
    
    # Step 2: Remove data with category 'none' (case insensitive)
    print("\n" + "-" * 70)
    print("Step 1: Remove data with category 'none'")
    print("-" * 70)
    
    none_count = 0
    categories_to_remove = []
    for cat in category_data:
        # Handle None type or string "none" (case insensitive)
        if cat is None or (isinstance(cat, str) and cat.lower() == 'none'):
            none_count += len(category_data[cat])
            categories_to_remove.append(cat)
    
    for cat in categories_to_remove:
        del category_data[cat]
    
    print(f"  Removed 'none' category data: {none_count} entries")
    
    # Step 3: Remove categories with fewer than 10 instances
    print("\n" + "-" * 70)
    print("Step 2: Remove categories with fewer than 10 instances")
    print("-" * 70)
    
    small_categories = [(cat, len(items)) for cat, items in category_data.items() if len(items) < 10]
    small_count = sum(count for _, count in small_categories)
    
    for cat, count in small_categories:
        print(f"  Removed category '{cat}': {count} entries")
        del category_data[cat]
    
    print(f"  Removed {len(small_categories)} small categories, total {small_count} entries")
    
    # Step 4: Balance the largest category
    print("\n" + "-" * 70)
    print("Step 3: Balance category data volume")
    print("-" * 70)
    
    # Sort categories by count
    sorted_categories = sorted(category_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    if len(sorted_categories) >= 2:
        a1_name, a1_data = sorted_categories[0]
        a2_name, a2_data = sorted_categories[1]
        a1_count = len(a1_data)
        a2_count = len(a2_data)
        max_a1_count = a2_count * 3
        
        print(f"  Largest category A1: '{a1_name}' = {a1_count} entries")
        print(f"  Second largest category A2: '{a2_name}' = {a2_count} entries")
        print(f"  A1 max allowed count (A2 × 3): {max_a1_count}")
        
        if a1_count > max_a1_count:
            print(f"\n  A1 count ({a1_count}) exceeds limit ({max_a1_count}), filtering required...")
            
            # Separate high priority data (video_dependency_score=10 and audio_dependency_score=10)
            high_priority = []
            normal_priority = []
            
            for item in a1_data:
                scores = get_scores(item)
                if scores['video_dependency_score'] == 10 and scores['audio_dependency_score'] == 10:
                    high_priority.append(item)
                else:
                    normal_priority.append(item)
            
            print(f"  High priority data (video=10, audio=10): {len(high_priority)} entries")
            print(f"  Normal priority data: {len(normal_priority)} entries")
            
            if len(high_priority) >= max_a1_count:
                # High priority data exceeds limit, sort by composite_score and take top max_a1_count
                print(f"  High priority data exceeds limit, sorting by composite_score and taking top {max_a1_count}")
                high_priority.sort(key=lambda x: get_scores(x)['composite_score'], reverse=True)
                final_a1_data = high_priority[:max_a1_count]
            else:
                # Keep all high priority, then sort by composite_score to fill remaining
                remaining_slots = max_a1_count - len(high_priority)
                normal_priority.sort(key=lambda x: get_scores(x)['composite_score'], reverse=True)
                final_a1_data = high_priority + normal_priority[:remaining_slots]
                print(f"  Keeping all high priority ({len(high_priority)}) + top {remaining_slots} normal priority")
            
            category_data[a1_name] = final_a1_data
            print(f"  A1 final count: {len(final_a1_data)} entries")
        else:
            print(f"\n  A1 count ({a1_count}) does not exceed limit ({max_a1_count}), no adjustment needed")
    else:
        print("  Fewer than 2 categories, skipping balance step")
    
    # Calculate final results
    print("\n" + "=" * 70)
    print("Final category distribution:")
    print("=" * 70)
    
    final_data = []
    final_counts = {}
    
    for cat, items in sorted(category_data.items(), key=lambda x: len(x[1]), reverse=True):
        final_counts[cat] = len(items)
        final_data.extend(items)
        print(f"  {cat}: {len(items)}")
    
    print(f"\nFinal data count: {len(final_data)}")
    print(f"Final category count: {len(final_counts)}")
    print(f"Retention ratio: {len(final_data) / total * 100:.2f}%")
    
    # Save results
    print(f"\nSaving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for data in final_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
