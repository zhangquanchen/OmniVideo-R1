#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract multiple choice QA data from LLaVA-Video-178K processed file.
This script filters entries where Type contains 'mc'.
"""

import json
from tqdm import tqdm
import os

def count_lines(filepath):
    """Count total lines in file for progress bar."""
    print(f"Counting lines in {filepath}...")
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def extract_mc(input_file, output_file):
    """
    Extract entries with 'mc' in Type field from jsonl file.
    
    Args:
        input_file: Path to input jsonl file
        output_file: Path to output jsonl file
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return
    
    # Count total lines for progress bar
    total_lines = count_lines(input_file)
    
    extracted_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"Processing {input_file}...")
    print(f"Output will be saved to {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(tqdm(infile, total=total_lines, desc="Processing"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Check if Type field contains 'mc'
                if 'Type' in data and 'mc' in data['Type']:
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    extracted_count += 1
                else:
                    skipped_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"\nError parsing line {line_num}: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"\nUnexpected error at line {line_num}: {e}")
                error_count += 1
                continue
    
    # Print summary
    print("\n" + "="*60)
    print("Extraction Complete!")
    print("="*60)
    print(f"Total lines processed: {total_lines}")
    print(f"Entries extracted (with mc): {extracted_count}")
    print(f"Entries skipped (without mc): {skipped_count}")
    print(f"Errors encountered: {error_count}")
    print(f"\nOutput saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    # Define input and output paths
    input_file = "/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/LLaVA-Video-178K_processed_video_audio.jsonl"
    output_file = "/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/LLaVA-Video-178K_processed_video_audio_mc_only.jsonl"
    
    # Extract mc entries
    extract_mc(input_file, output_file)

