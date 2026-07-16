#!/usr/bin/env python3
"""
Replace paths in JSONL file.

Usage:
    python replace_paths.py --input input.jsonl --output output.jsonl --old_path /old/path --new_path /new/path
"""

import json
import argparse


def replace_paths(text, replacements):
    """Replace all paths in text"""
    for old_path, new_path in replacements.items():
        text = text.replace(old_path, new_path)
    return text


def main():
    parser = argparse.ArgumentParser(description="Replace paths in JSONL file")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--old_path", required=True, help="Old path")
    parser.add_argument("--new_path", required=True, help="New path")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    # Define replacement rules
    replacements = {
        args.old_path: args.new_path
    }
    
    count = 0
    replaced_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            count += 1
            original_line = line
            # Perform string replacement directly on the entire line
            new_line = replace_paths(line, replacements)
            
            if new_line != original_line:
                replaced_count += 1
            
            fout.write(new_line)
            
            if count % 10000 == 0:
                print(f"Processed {count} lines, replaced {replaced_count} lines...")
    
    print(f"\nDone!")
    print(f"Total processed: {count} lines")
    print(f"Replaced: {replaced_count} lines")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()
