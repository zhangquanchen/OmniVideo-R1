#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for QueryIntentORM class.

Usage:
    python test_query_intent.py --video /path/to/your/video.mp4
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnivideo_r1 import QueryIntentORM


def main():
    parser = argparse.ArgumentParser(description="Test QueryIntentORM")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--model", type=str, default="Qwen3-VL-235B-Instruct", help="Model name")
    parser.add_argument("--video", type=str, default="./test_video.mp4", help="Path to test video file")
    args = parser.parse_args()
    # Initialize ORM
    orm = QueryIntentORM(
        base_url=args.api_url,
        judge_model=args.model,
        timeout=120,
        temp_dir="/tmp/video_segments_test"
    )
    
    # ========== Test Data ==========
    # completions: model outputs with <time>...</time><caption>...</caption> format
    completions = [
        """<time>0.0-2.0</time><caption>The video begins with an opening scene showing the main subject</caption>
<time>2.0-4.0</time><caption>The scene transitions to show more details of the action</caption>
<thinking>Based on the video segments, I observed the key events happening in the video.</thinking>
<answer>The video shows a sequence of events with clear visual elements.</answer>"""
    ]
    
    # solution: ground truth (not used by this ORM, can be None)
    solution = ["The video shows a sequence of events with clear visual elements."]
    
    # problem: the question asked about the video
    problem = ["<video>What is happening in this video? Please describe the main events."]
    
    # videos: list of video paths for each sample
    videos = [[args.video]]
    
    try:
        rewards = orm(
            completions=completions,
            solution=solution,
            problem=problem,
            videos=videos
        )
        print("\n" + "=" * 60)
        print(f"Rewards: {rewards}")
        return 0
        
    except Exception as e:
        print(f"✗ FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
