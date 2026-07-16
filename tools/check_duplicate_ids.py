#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check duplicate IDs in JSONL file

Usage:
    python check_duplicate_ids.py <file_path>
"""

import json
from collections import Counter
import sys

def check_duplicate_ids(file_path):
    """Check duplicate IDs in file"""
    ids = []
    
    print(f"Reading file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                id_val = data.get('id', f'no_id_line_{line_num}')
                ids.append(id_val)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} JSON parsing error: {e}")
    
    print(f"\nTotal records: {len(ids)}")
    print(f"Unique IDs: {len(set(ids))}")
    print(f"Duplicate IDs: {len(ids) - len(set(ids))}")
    
    # Find duplicate IDs
    id_counts = Counter(ids)
    duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}
    
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate IDs:")
        print("="*80)
        # Sort by count
        for id_val, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {id_val}: appears {count} times")
        
        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more duplicate IDs not shown")
    else:
        print("\n✓ No duplicate IDs found")
    
    return duplicates

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("Usage: python check_duplicate_ids.py <file_path>")
        print("Example: python check_duplicate_ids.py ./data/train.jsonl")
        sys.exit(1)
    
    check_duplicate_ids(file_path)
