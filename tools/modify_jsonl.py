import json
import os
from tqdm import tqdm

input_file = '/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/LLaVA-Video-178K_processed_video_audio_mc_only.jsonl'
output_file = '/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/LLaVA-Video-178K_processed_video_audio_mc_only_with_prompt.jsonl'

suffix_text = "\nFirst output the thinking process in <think> </think> tags and then output the final answer (option) in <answer> </answer> tags."

# Count total lines first for progress bar
print("Counting total lines...")
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

print(f"Total lines: {total_lines}")
print("Processing...")

modified_count = 0
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in tqdm(infile, total=total_lines):
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            
            # Modify messages with role="user"
            if 'messages' in data and isinstance(data['messages'], list):
                for msg in data['messages']:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        if 'content' in msg:
                            # Check if suffix already exists
                            if not msg['content'].endswith(suffix_text):
                                msg['content'] = msg['content'] + suffix_text
                                modified_count += 1
            
            # Modify problem field if exists
            if 'problem' in data and isinstance(data['problem'], str):
                if not data['problem'].endswith(suffix_text):
                    data['problem'] = data['problem'] + suffix_text
                    modified_count += 1
            
            # Write modified data
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Line: {line[:100]}...")
            continue

print(f"\nProcessing complete!")
print(f"Modified {modified_count} fields")
print(f"Output saved to: {output_file}")
print("Done!")

