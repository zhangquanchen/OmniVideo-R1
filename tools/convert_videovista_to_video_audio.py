#!/usr/bin/env python3
"""
Convert VideoVista dataset format to multimodal format with video and audio.

Changes:
1. Rename 'video' field to 'videos'
2. Add 'audios' field with audio paths (derived from video paths)
3. Replace '<video>' with '<video><audio>' in solution and messages
4. Extract audio from video files using imageio-ffmpeg

Requirements:
    pip install imageio-ffmpeg tqdm
"""

import json
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def convert_video_path_to_audio_path(video_path):
    """
    Convert video path to audio path.
    
    Example:
        /apdcephfs_hldy/share_303558466/jankinchen/data/VideoVista_Train/VideoVista-Train-videos/Rl0M4-VNpe4.120t300.0.mp4
        -> 
        /apdcephfs_hldy/share_303558466/jankinchen/data/VideoVista_Train/VideoVista-Train-audios/Rl0M4-VNpe4.120t300.0.mp3
    """
    # Parse the path
    path_obj = Path(video_path)
    
    # Replace 'videos' directory with 'audios' and change extension to .mp3
    audio_path = str(path_obj).replace('/VideoVista-Train-videos/', '/VideoVista-Train-audios/')
    audio_path = audio_path.replace('.mp4', '.mp3')
    
    return audio_path


def replace_video_tag_in_content(content):
    """Replace <video> with <video><audio> in content string."""
    if content and isinstance(content, str):
        return content.replace('<video>', '<video><audio>')
    return content


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


def process_jsonl_line(line):
    """Process a single line from the JSONL file."""
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse line: {line[:100]}...")
        return None
    
    # 1. Rename 'video' to 'videos'
    if 'video' in data:
        data['videos'] = data.pop('video')
    
    # 2. Add 'audios' field
    if 'videos' in data and isinstance(data['videos'], list):
        data['audios'] = [convert_video_path_to_audio_path(v) for v in data['videos']]
    
    # 3. Replace <video> with <video><audio> in messages
    if 'messages' in data:
        for message in data['messages']:
            if 'content' in message:
                message['content'] = replace_video_tag_in_content(message['content'])
    
    # Replace in 'problem' field if exists
    if 'problem' in data:
        data['problem'] = replace_video_tag_in_content(data['problem'])
    
    # Replace in 'solution' field if exists
    if 'solution' in data:
        data['solution'] = replace_video_tag_in_content(data['solution'])
    
    return data


def main():
    input_file = '/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/VideoVista_Train_processed.jsonl'
    output_file = '/apdcephfs_hldy/share_303558466/jankinchen/ominithinker/data/VideoVista_Train_processed_video_audio.jsonl'
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    # Step 1: Collect all unique video paths
    print("\n" + "="*60)
    print("STEP 1: Collecting unique video paths...")
    print("="*60)
    
    video_paths = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading videos"):
            try:
                data = json.loads(line.strip())
                if 'video' in data:
                    for video_path in data['video']:
                        video_paths.add(video_path)
            except json.JSONDecodeError:
                continue
    
    video_paths = sorted(list(video_paths))
    print(f"Found {len(video_paths)} unique videos")
    
    # Step 2: Extract audio from videos
    print("\n" + "="*60)
    print("STEP 2: Extracting audio from videos...")
    print("="*60)
    
    video_audio_pairs = []
    for video_path in video_paths:
        audio_path = convert_video_path_to_audio_path(video_path)
        video_audio_pairs.append((video_path, audio_path))
    
    # Check existing
    existing_count = sum(1 for _, audio_path in video_audio_pairs if os.path.exists(audio_path))
    print(f"Audio files already exist: {existing_count}/{len(video_audio_pairs)}")
    print(f"Audio files to extract: {len(video_audio_pairs) - existing_count}")
    
    # Extract audio using multiple workers
    num_workers = min(multiprocessing.cpu_count(), 8)
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
        print(f"\n⚠️  First 5 failed extractions:")
        for video_path, error_msg in failed_videos[:5]:
            print(f"  - {os.path.basename(video_path)}: {error_msg}")
    
    # Step 3: Process JSONL file
    print("\n" + "="*60)
    print("STEP 3: Converting JSONL format...")
    print("="*60)
    
    # Count total lines
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines to process: {total_lines}")
    
    processed_count = 0
    line_skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, total=total_lines, desc="Processing JSONL"):
            processed_data = process_jsonl_line(line)
            
            if processed_data is not None:
                outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                processed_count += 1
            else:
                line_skipped_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nAudio Extraction:")
    print(f"  Total unique videos: {len(video_audio_pairs)}")
    print(f"  Successfully extracted: {success_count}")
    print(f"  Already existed: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"\nJSONL Conversion:")
    print(f"  Total lines processed: {processed_count}")
    print(f"  Lines skipped: {line_skipped_count}")
    print(f"\nOutput file: {output_file}")
    print("="*60)
    
    # Show sample
    print("\nSample of first converted entry:")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

