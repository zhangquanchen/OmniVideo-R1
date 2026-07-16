#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for TimeCaptionThinkAnswerFormatORM
"""
import re
from typing import List


class TimeCaptionThinkAnswerFormatORM:
    """
    Reward function that checks if the completion follows the format:
    Multiple <time>start-end</time><caption>description</caption> segments
    followed by <thinking>...</thinking><answer>...</answer>
    
    Every <time> must have a corresponding <caption> after it.
    Every <caption> must have a corresponding <time> before it.
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            reward = 0.0
            num_time_caption_pairs = 0
            num_total_times = 0
            num_total_captions = 0
            has_think_answer = False
            try:
                # Pattern for <time>start-end</time><caption>...</caption> pairs
                time_caption_pattern = r'<time>\s*\d+\.?\d*\s*-\s*\d+\.?\d*\s*</time>\s*<caption>.*?</caption>'
                
                # Pattern to find all <time>...</time> tags
                time_only_pattern = r'<time>.*?</time>'
                
                # Pattern to find all <caption>...</caption> tags
                caption_only_pattern = r'<caption>.*?</caption>'
                
                # Pattern for <thinking>...</thinking><answer>...</answer>
                think_answer_pattern = r'<thinking>.*?</thinking>\s*<answer>.*?</answer>'
                
                # Find all time-caption pairs
                time_caption_matches = re.findall(time_caption_pattern, content, re.DOTALL)
                num_time_caption_pairs = len(time_caption_matches)
                
                # Find all time tags
                time_matches = re.findall(time_only_pattern, content, re.DOTALL)
                num_total_times = len(time_matches)
                
                # Find all caption tags
                caption_matches = re.findall(caption_only_pattern, content, re.DOTALL)
                num_total_captions = len(caption_matches)
                
                # Check if thinking-answer pattern exists
                think_answer_match = re.search(think_answer_pattern, content, re.DOTALL)
                has_think_answer = bool(think_answer_match)
                
                # Check if all times and captions are properly paired
                all_properly_paired = (num_time_caption_pairs == num_total_times == num_total_captions)
                
                # Only give reward if all conditions are met
                if num_time_caption_pairs >= 1 and all_properly_paired and has_think_answer:
                    reward = 1.0
                    
            except Exception:
                pass
            
            print("--------------------------------")
            print(f"TimeCaptionThinkAnswer Format Reward: {reward}")
            rewards.append(reward)
        return rewards


def test_format_reward():
    """Test the TimeCaptionThinkAnswerFormatORM with various inputs"""
    
    orm = TimeCaptionThinkAnswerFormatORM()
    
    # Test Case 1: Standalone time without caption (should get 0.0)
    test1 = """<time>5.2-6.8</time><caption>The person paints a sun on a white canvas.</caption>
<time>5.2-6.8</time><caption>The person paints a sun on a white canvas.</caption>
<thinking>Let me think.</thinking>
<answer>Painting a sun on a canvas</answer>"""
    
    print("=" * 60)
    print("Test 1: Standalone time without caption (should be 0.0)")
    print("=" * 60)
    result = orm([test1])
    print(f"Result: {result[0]}")
    print()


if __name__ == "__main__":
    test_format_reward()
