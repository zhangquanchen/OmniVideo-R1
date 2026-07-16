#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to replace old paths with new paths in a JSONL file.

Usage:
    python replace_paths.py --input input.jsonl --output output.jsonl --old_path /old/path --new_path /new/path
"""

import json
import os
import argparse
from pathlib import Path

def replace_paths_in_value(value, old_path, new_path):
    """
    Recursively replace old path with new path in any value.
    """
    if isinstance(value, str):
        return value.replace(old_path, new_path)
    elif isinstance(value, dict):
        return {k: replace_paths_in_value(v, old_path, new_path) for k, v in value.items()}
    elif isinstance(value, list):
        return [replace_paths_in_value(item, old_path, new_path) for item in value]
    else:
        return value

def main():
    parser = argparse.ArgumentParser(description="Replace paths in JSONL file")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--old_path", required=True, help="Old path to replace")
    parser.add_argument("--new_path", required=True, help="New path to use")
    parser.add_argument("--backup", "-b", help="Backup file path (optional)")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    old_path = args.old_path
    new_path = args.new_path
    backup_file = args.backup
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Create backup if specified
    if backup_file:
        print(f"Creating backup: {backup_file}")
        os.system(f'cp "{input_file}" "{backup_file}"')
    
    # Process the file
    print(f"Processing file: {input_file}")
    print(f"Replacing: {old_path}")
    print(f"With: {new_path}")
    processed_lines = 0
    replaced_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON
                data = json.loads(line)
                
                # Replace paths in the entire data structure
                updated_data = replace_paths_in_value(data, old_path, new_path)
                
                # Check if any replacement was made
                if json.dumps(data) != json.dumps(updated_data):
                    replaced_count += 1
                
                # Write updated JSON
                f_out.write(json.dumps(updated_data, ensure_ascii=False) + '\n')
                processed_lines += 1
                
                if processed_lines % 1000 == 0:
                    print(f"Processed {processed_lines} lines, {replaced_count} lines with replacements...")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                # Write original line if parsing fails
                f_out.write(line + '\n')
    
    print(f"\nCompleted!")
    print(f"Total lines processed: {processed_lines}")
    print(f"Lines with replacements: {replaced_count}")
    print(f"Output file: {output_file}")
    if backup_file:
        print(f"Backup file: {backup_file}")

if __name__ == "__main__":
    main()
