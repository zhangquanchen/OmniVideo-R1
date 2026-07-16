#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process VideoVista_Train.json file
- Convert video field to an array with full path prefix
- Convert conversations to messages format (human->user, gpt->assistant)
- Replace <image> with <video>
- Add problem and solution fields
- Filter out records where conversation values are not strings
- Save as jsonl format
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def validate_conversation_value(conv: Dict[str, Any]) -> bool:
    """
    Check if the conversation value field is a string.
    Returns True if valid, False otherwise.
    """
    return isinstance(conv.get('value'), str)


def convert_conversations_to_messages(conversations: List[Dict[str, Any]]) -> Optional[tuple]:
    """
    Convert conversations format to messages format.
    Returns (messages, problem, solution) tuple if successful, None if invalid.
    
    Args:
        conversations: List of conversation dicts with 'from' and 'value' keys
        
    Returns:
        Tuple of (messages_list, problem_str, solution_str) or None if invalid
    """
    # Validate all conversation values are strings
    for conv in conversations:
        if not validate_conversation_value(conv):
            return None
    
    messages = []
    problem = None
    solution = None
    
    for conv in conversations:
        role_map = {
            'human': 'user',
            'gpt': 'assistant'
        }
        
        from_role = conv.get('from')
        value = conv.get('value', '')
        
        # Replace <image> with <video>
        value = value.replace('<image>', '<video>')
        
        # Map role
        role = role_map.get(from_role)
        if role is None:
            continue
            
        messages.append({
            'role': role,
            'content': value
        })
        
        # Extract problem (first user message) and solution (first assistant message)
        if role == 'user' and problem is None:
            problem = value
        elif role == 'assistant' and solution is None:
            solution = value
    
    return messages, problem, solution


def process_record(record: Dict[str, Any], video_prefix: str) -> Optional[Dict[str, Any]]:
    """
    Process a single record.
    Returns the processed record or None if should be filtered out.
    """
    # Convert video field to array with prefix
    video_filename = record.get('video')
    if not video_filename:
        return None
    
    video_path = f"{video_prefix}/{video_filename}"
    
    # Convert conversations to messages
    conversations = record.get('conversations', [])
    conversion_result = convert_conversations_to_messages(conversations)
    
    if conversion_result is None:
        # Invalid conversation values, skip this record
        return None
    
    messages, problem, solution = conversion_result
    
    # Build the new record
    new_record = {
        'id': record.get('id'),
        'video': [video_path],
        'Type': record.get('Type'),
        'messages': messages,
        'problem': problem,
        'solution': solution
    }
    
    return new_record


def process_file(input_file: str, output_file: str, video_prefix: str):
    """
    Process the entire JSON file and save as JSONL.
    """
    print(f"Reading input file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    if not isinstance(data, list):
        print("Error: Input JSON must be an array")
        sys.exit(1)
    
    print(f"Total records in input: {len(data)}")
    
    processed_count = 0
    filtered_count = 0
    
    print(f"Writing output file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, record in enumerate(data):
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} records...")
            
            processed_record = process_record(record, video_prefix)
            
            if processed_record is None:
                filtered_count += 1
                continue
            
            # Write as JSONL (one JSON object per line)
            json_line = json.dumps(processed_record, ensure_ascii=False)
            f.write(json_line + '\n')
            processed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Total records processed: {processed_count}")
    print(f"Total records filtered out: {filtered_count}")
    print(f"Output saved to: {output_file}")


def main():
    # Configuration
    input_file = "/apdcephfs_hldy/share_303558466/jankinchen/data/VideoVista_Train/VideoVista_Train.json"
    output_file = "/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/VideoVista_Train_processed.jsonl"
    video_prefix = "/apdcephfs_hldy/share_303558466/jankinchen/data/VideoVista_Train/VideoVista-Train-videos"
    
    # Process the file
    process_file(input_file, output_file, video_prefix)


if __name__ == "__main__":
    main()

