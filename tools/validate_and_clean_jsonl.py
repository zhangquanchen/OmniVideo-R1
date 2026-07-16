#!/usr/bin/env python3
"""
Validate and clean LLaVA-Video-178K processed JSONL file.

This script:
1. Reads each line from the JSONL file
2. Validates data format (required fields, data types)
3. Checks if video/audio file paths exist
4. Removes invalid entries and prints them
5. Writes valid entries to a new file
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


# Configuration
INPUT_FILE = "/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/LLaVA-Video-178K_processed_video_audio.jsonl"
OUTPUT_FILE = "/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/LLaVA-Video-178K_processed_video_audio_cleaned.jsonl"
BACKUP_FILE = "/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/LLaVA-Video-178K_processed_video_audio_backup.jsonl"


def count_lines(file_path):
    """Count total lines in file for progress bar."""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def validate_entry_format(entry, line_num):
    """
    Validate that an entry has all required fields and correct data types.
    
    Args:
        entry: Dictionary to validate
        line_num: Line number in file (for error messages)
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    # Required fields
    required_fields = ["id", "Type", "messages", "problem", "solution", "videos", "audios"]
    
    # Check if entry is a dictionary
    if not isinstance(entry, dict):
        return False, f"Entry is not a dictionary"
    
    # Check required fields exist
    for field in required_fields:
        if field not in entry:
            return False, f"Missing required field: {field}"
    
    # Validate data types
    if not isinstance(entry["messages"], list):
        return False, f"'messages' must be a list"
    
    if not isinstance(entry["videos"], list):
        return False, f"'videos' must be a list"
    
    if not isinstance(entry["audios"], list):
        return False, f"'audios' must be a list"
    
    # Check that lists are not empty
    if len(entry["videos"]) == 0:
        return False, f"'videos' list is empty"
    
    if len(entry["audios"]) == 0:
        return False, f"'audios' list is empty"
    
    # Validate messages format
    for i, msg in enumerate(entry["messages"]):
        if not isinstance(msg, dict):
            return False, f"messages[{i}] is not a dictionary"
        if "role" not in msg or "content" not in msg:
            return False, f"messages[{i}] missing 'role' or 'content'"
        if msg["role"] not in ["user", "assistant", "system"]:
            return False, f"messages[{i}] has invalid role: {msg['role']}"
    
    # Check that problem and solution are strings
    if entry["problem"] is not None and not isinstance(entry["problem"], str):
        return False, f"'problem' must be a string or null"
    
    if entry["solution"] is not None and not isinstance(entry["solution"], str):
        return False, f"'solution' must be a string or null"
    
    return True, None


def check_files_exist(entry):
    """
    Check if all video and audio files in entry exist.
    
    Args:
        entry: Dictionary with 'videos' and 'audios' fields
        
    Returns:
        tuple: (all_exist: bool, missing_files: list)
    """
    missing_files = []
    
    # Check videos
    for video_path in entry.get("videos", []):
        if not os.path.exists(video_path):
            missing_files.append(("video", video_path))
    
    # Check audios
    for audio_path in entry.get("audios", []):
        if not os.path.exists(audio_path):
            missing_files.append(("audio", audio_path))
    
    return len(missing_files) == 0, missing_files


def main():
    print("="*80)
    print("JSONL Validation and Cleaning Tool")
    print("="*80)
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Backup file: {BACKUP_FILE}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ Error: Input file does not exist: {INPUT_FILE}")
        return
    
    # Count total lines
    print("\n" + "="*80)
    print("Counting total entries...")
    print("="*80)
    total_lines = count_lines(INPUT_FILE)
    print(f"Total entries to process: {total_lines:,}")
    
    # Statistics
    stats = {
        "total": 0,
        "valid": 0,
        "invalid_format": 0,
        "missing_files": 0,
        "parse_error": 0
    }
    
    invalid_entries = []
    
    # Process file
    print("\n" + "="*80)
    print("Processing and validating entries...")
    print("="*80)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(tqdm(infile, total=total_lines, desc="Validating"), start=1):
            stats["total"] += 1
            
            # Skip empty lines
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON
                entry = json.loads(line)
                
                # Validate format
                is_valid_format, format_error = validate_entry_format(entry, line_num)
                if not is_valid_format:
                    stats["invalid_format"] += 1
                    invalid_entries.append({
                        "line": line_num,
                        "id": entry.get("id", "unknown"),
                        "reason": "Invalid format",
                        "details": format_error
                    })
                    continue
                
                # Check if files exist
                files_exist, missing_files = check_files_exist(entry)
                if not files_exist:
                    stats["missing_files"] += 1
                    invalid_entries.append({
                        "line": line_num,
                        "id": entry.get("id", "unknown"),
                        "reason": "Missing files",
                        "details": missing_files
                    })
                    continue
                
                # Entry is valid - write to output
                stats["valid"] += 1
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                stats["parse_error"] += 1
                invalid_entries.append({
                    "line": line_num,
                    "id": "unknown",
                    "reason": "JSON parse error",
                    "details": str(e)
                })
            except Exception as e:
                stats["parse_error"] += 1
                invalid_entries.append({
                    "line": line_num,
                    "id": "unknown",
                    "reason": "Unexpected error",
                    "details": str(e)
                })
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Total entries processed: {stats['total']:,}")
    print(f"  ✅ Valid entries: {stats['valid']:,} ({stats['valid']/stats['total']*100:.2f}%)")
    print(f"  ❌ Invalid format: {stats['invalid_format']:,}")
    print(f"  ❌ Missing files: {stats['missing_files']:,}")
    print(f"  ❌ Parse errors: {stats['parse_error']:,}")
    print(f"  Total removed: {stats['invalid_format'] + stats['missing_files'] + stats['parse_error']:,}")
    
    # Print invalid entries
    if invalid_entries:
        print("\n" + "="*80)
        print(f"INVALID ENTRIES (Total: {len(invalid_entries)})")
        print("="*80)
        
        # Group by reason
        by_reason = {}
        for entry in invalid_entries:
            reason = entry["reason"]
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(entry)
        
        for reason, entries in by_reason.items():
            print(f"\n{reason} ({len(entries)} entries):")
            print("-" * 80)
            # Show first 20 of each type
            for entry in entries[:20]:
                print(f"  Line {entry['line']}, ID: {entry['id']}")
                print(f"  Details: {entry['details']}")
                print()
            
            if len(entries) > 20:
                print(f"  ... and {len(entries) - 20} more entries of this type")
                print()
    
    # Save detailed log
    log_file = OUTPUT_FILE.replace('.jsonl', '_validation_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "input_file": INPUT_FILE,
            "output_file": OUTPUT_FILE,
            "statistics": stats,
            "invalid_entries": invalid_entries
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Detailed log saved to: {log_file}")
    print(f"\n✅ Cleaned data saved to: {OUTPUT_FILE}")
    
    # Suggest backup
    if stats["valid"] < stats["total"]:
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Review the invalid entries above")
        print("2. If satisfied, backup the original file:")
        print(f"   mv {INPUT_FILE} {BACKUP_FILE}")
        print("3. Replace with cleaned file:")
        print(f"   mv {OUTPUT_FILE} {INPUT_FILE}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

