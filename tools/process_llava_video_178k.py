#!/usr/bin/env python3
"""
Process LLaVA-Video-178K dataset to VideoVista format with video and audio.

This script:
1. Reads all JSON files in LLaVA-Video-178K subdirectories
2. Extracts audio from video files
3. Converts conversations format to messages format
4. Generates a JSONL file with required fields: messages, problem, solution, videos, audios

Requirements:
    pip install imageio-ffmpeg tqdm
"""

import json
import os
import subprocess
import glob
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Configuration
BASE_DATA_DIR = "/apdcephfs_hldy/share_303558466/jankinchen/data/LLaVA-Video-178K"
OUTPUT_FILE = "/apdcephfs_hldy/share_303558466/jankinchen/data/LLaVA-Video-178K_processed_video_audio.jsonl"
AUDIO_OUTPUT_BASE = "/apdcephfs_hldy/share_303558466/jankinchen/data/LLaVA-Video-178K-audios"

def convert_video_path_to_audio_path(video_path):
    """
    Convert absolute video path to absolute audio path.
    
    Example:
        /apdcephfs_hldy/.../LLaVA-Video-178K/0_30_s_academic_v0_1/academic_source/Charades/RW587.mp4
        ->
        /apdcephfs_hldy/.../LLaVA-Video-178K-audios/0_30_s_academic_v0_1/academic_source/Charades/RW587.mp3
    """
    # Extract the relative path from BASE_DATA_DIR
    if BASE_DATA_DIR in video_path:
        rel_path = video_path.replace(BASE_DATA_DIR + "/", "")
        audio_path = os.path.join(AUDIO_OUTPUT_BASE, rel_path)
        audio_path = audio_path.replace('.mp4', '.mp3')
        return audio_path
    else:
        # Fallback: just replace .mp4 with .mp3 and use audio base directory
        video_filename = os.path.basename(video_path)
        audio_filename = video_filename.replace('.mp4', '.mp3')
        return os.path.join(AUDIO_OUTPUT_BASE, audio_filename)


def extract_audio_from_video(video_path, audio_path):
    """
    Extract audio from video file using imageio-ffmpeg.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file
    
    Returns:
        tuple: (success: bool, video_path: str, error_msg: str)
    """
    try:
        # Check if video exists
        if not os.path.exists(video_path):
            return False, video_path, f"Video file not found"
        
        # Create output directory if not exists
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        # Skip if audio already exists
        if os.path.exists(audio_path):
            return True, video_path, "exists"
        
        # Get ffmpeg executable from imageio-ffmpeg
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Extract audio using ffmpeg
        cmd = [
            ffmpeg_exe,
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-q:a', '2',  # High quality
            '-y',  # Overwrite
            audio_path
        ]
        
        # Run ffmpeg (suppress output)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            return True, video_path, None
        else:
            # Check if it's because there's no audio track
            if 'does not contain any stream' in result.stderr or 'Output file is empty' in result.stderr:
                return False, video_path, "No audio track"
            return False, video_path, f"ffmpeg error"
            
    except Exception as e:
        return False, video_path, f"Exception: {str(e)}"


def convert_conversations_to_messages(conversations):
    """
    Convert LLaVA conversations format to messages format.
    Also replace <image> with <video><audio>.
    
    Args:
        conversations: List of dicts with 'from' and 'value' keys
        
    Returns:
        List of dicts with 'role' and 'content' keys
    """
    messages = []
    for conv in conversations:
        role = "user" if conv["from"] == "human" else "assistant"
        content = conv["value"]
        # Replace <image> with <video><audio>
        content = content.replace("<image>", "<video><audio>")
        messages.append({"role": role, "content": content})
    return messages


def extract_problem_solution(conversations):
    """
    Extract problem and solution from conversations.
    Problem is the first human message, solution is the first gpt message.
    
    Args:
        conversations: List of dicts with 'from' and 'value' keys
        
    Returns:
        tuple: (problem: str, solution: str)
    """
    problem = None
    solution = None
    
    for conv in conversations:
        if conv["from"] == "human" and problem is None:
            problem = conv["value"].replace("<image>", "<video><audio>")
        elif conv["from"] == "gpt" and solution is None:
            solution = conv["value"]
            break  # We have both now
    
    return problem, solution


def process_json_file(json_file_path, subdirectory_name):
    """
    Process a single JSON file and return list of processed entries.
    
    Args:
        json_file_path: Path to the JSON file
        subdirectory_name: Name of the subdirectory (used as Type)
        
    Returns:
        List of processed data entries
    """
    processed_entries = []
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine the type from the JSON file name
        json_filename = os.path.basename(json_file_path)
        type_name = json_filename.replace('.json', '')
        
        for entry in data:
            # Build absolute video path
            video_rel_path = entry.get("video", "")
            video_abs_path = os.path.join(BASE_DATA_DIR, subdirectory_name, video_rel_path)
            
            # Get audio path
            audio_abs_path = convert_video_path_to_audio_path(video_abs_path)
            
            # Convert conversations to messages
            conversations = entry.get("conversations", [])
            messages = convert_conversations_to_messages(conversations)
            
            # Extract problem and solution
            problem, solution = extract_problem_solution(conversations)
            
            # Build output entry
            output_entry = {
                "id": entry.get("id", ""),
                "Type": type_name,
                "messages": messages,
                "problem": problem,
                "solution": solution,
                "videos": [video_abs_path],
                "audios": [audio_abs_path]
            }
            
            # Add optional fields if they exist
            if "data_source" in entry:
                output_entry["data_source"] = entry["data_source"]
            
            processed_entries.append(output_entry)
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
    
    return processed_entries


def find_all_json_files():
    """
    Find all JSON files in subdirectories of BASE_DATA_DIR.
    Exclude gpt4o_qa_prompt directory.
    
    Returns:
        List of tuples: (json_file_path, subdirectory_name)
    """
    json_files = []
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(BASE_DATA_DIR) 
               if os.path.isdir(os.path.join(BASE_DATA_DIR, d)) 
               and d != 'gpt4o_qa_prompt']
    
    for subdir in subdirs:
        subdir_path = os.path.join(BASE_DATA_DIR, subdir)
        # Find all JSON files in this subdirectory
        pattern = os.path.join(subdir_path, "*.json")
        for json_file in glob.glob(pattern):
            json_files.append((json_file, subdir))
    
    return json_files


def main():
    print("="*80)
    print("LLaVA-Video-178K to VideoVista Format Converter")
    print("="*80)
    print(f"\nInput directory: {BASE_DATA_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Audio output directory: {AUDIO_OUTPUT_BASE}")
    
    # Step 1: Find all JSON files
    print("\n" + "="*80)
    print("STEP 1: Finding all JSON files...")
    print("="*80)
    
    json_files = find_all_json_files()
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file, subdir in json_files[:10]:  # Show first 10
        print(f"  - {subdir}/{os.path.basename(json_file)}")
    if len(json_files) > 10:
        print(f"  ... and {len(json_files) - 10} more")
    
    # Step 2: Process all JSON files
    print("\n" + "="*80)
    print("STEP 2: Processing JSON files...")
    print("="*80)
    
    all_entries = []
    video_paths = set()
    
    for json_file, subdir in tqdm(json_files, desc="Processing JSON files"):
        entries = process_json_file(json_file, subdir)
        all_entries.extend(entries)
        
        # Collect video paths
        for entry in entries:
            if 'videos' in entry:
                for video_path in entry['videos']:
                    video_paths.add(video_path)
    
    print(f"\nProcessed {len(all_entries)} data entries")
    print(f"Found {len(video_paths)} unique videos")
    
    # Step 3: Extract audio from videos
    print("\n" + "="*80)
    print("STEP 3: Extracting audio from videos...")
    print("="*80)
    
    video_audio_pairs = []
    for video_path in sorted(video_paths):
        audio_path = convert_video_path_to_audio_path(video_path)
        video_audio_pairs.append((video_path, audio_path))
    
    # Check existing
    existing_count = sum(1 for _, audio_path in video_audio_pairs if os.path.exists(audio_path))
    print(f"Audio files already exist: {existing_count}/{len(video_audio_pairs)}")
    print(f"Audio files to extract: {len(video_audio_pairs) - existing_count}")
    
    # Extract audio using multiple workers
    print(f"CPU count: {multiprocessing.cpu_count()}")
    num_workers = min(multiprocessing.cpu_count(), 128)
    print(f"Using {num_workers} parallel workers for audio extraction...")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_videos = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {
            executor.submit(extract_audio_from_video, video_path, audio_path): (video_path, audio_path)
            for video_path, audio_path in video_audio_pairs
        }
        
        with tqdm(total=len(video_audio_pairs), desc="Extracting audio") as pbar:
            for future in as_completed(future_to_video):
                video_path, audio_path = future_to_video[future]
                try:
                    success, vid_path, error_msg = future.result()
                    if success:
                        if error_msg == "exists":
                            skipped_count += 1
                        else:
                            success_count += 1
                    else:
                        failed_count += 1
                        failed_videos.append((vid_path, error_msg))
                except Exception as e:
                    failed_count += 1
                    failed_videos.append((video_path, str(e)))
                
                pbar.update(1)
    
    print(f"\nAudio extraction results:")
    print(f"  Successfully extracted: {success_count}")
    print(f"  Already existed: {skipped_count}")
    print(f"  Failed: {failed_count}")
    
    if failed_videos:
        print(f"\n⚠️  First 10 failed extractions:")
        for video_path, error_msg in failed_videos[:10]:
            print(f"  - {os.path.basename(video_path)}: {error_msg}")
    
    # Step 4: Write output JSONL file
    print("\n" + "="*80)
    print("STEP 4: Writing output JSONL file...")
    print("="*80)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in tqdm(all_entries, desc="Writing JSONL"):
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nWrote {len(all_entries)} entries to {OUTPUT_FILE}")
    
    # Final summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  JSON files processed: {len(json_files)}")
    print(f"  Data entries: {len(all_entries)}")
    print(f"  Unique videos: {len(video_audio_pairs)}")
    print(f"  Audio extracted: {success_count}")
    print(f"  Audio already existed: {skipped_count}")
    print(f"  Audio extraction failed: {failed_count}")
    print(f"\nOutput file: {OUTPUT_FILE}")
    print("="*80)
    
    # Show sample
    print("\nSample of first converted entry:")
    if all_entries:
        print(json.dumps(all_entries[0], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

