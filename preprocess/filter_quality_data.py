#!/usr/bin/env python3
"""
Data Quality Filtering Script

Features:
1. Check if all paths in videos and audios fields exist
2. Check if all field formats are uniform and compliant
3. Remove data where answer_accuracy_score is not 10 (full score)
4. Remove data where question_logic_score is less than 8
5. Remove data where composite_score is less than 7
6. Remove data where should_filter is true
7. Count quantity of each category

Usage:
    python filter_quality_data.py --input input.jsonl --output output.jsonl
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Tuple


class FilterStats:
    """Filter statistics class"""
    def __init__(self):
        self.total = 0
        self.missing_video_path = 0
        self.missing_audio_path = 0
        self.invalid_format = 0
        self.answer_accuracy_filtered = 0
        self.question_logic_filtered = 0
        self.composite_score_filtered = 0
        self.should_filter_filtered = 0
        self.kept = 0
        self.category_counts = defaultdict(int)
        
    def print_summary(self):
        print("\n" + "=" * 60)
        print("Data Filter Statistics Report")
        print("=" * 60)
        print(f"Original data count: {self.total}")
        print(f"\nFiltered statistics:")
        print(f"  - Video path not found: {self.missing_video_path}")
        print(f"  - Audio path not found: {self.missing_audio_path}")
        print(f"  - Invalid format: {self.invalid_format}")
        print(f"  - answer_accuracy_score != 10: {self.answer_accuracy_filtered}")
        print(f"  - question_logic_score < 8: {self.question_logic_filtered}")
        print(f"  - composite_score < 7: {self.composite_score_filtered}")
        print(f"  - should_filter = true: {self.should_filter_filtered}")
        print(f"\nKept data count: {self.kept}")
        if self.total > 0:
            print(f"Retention ratio: {self.kept / self.total * 100:.2f}%")
        
        print(f"\nCategory statistics (total {len(self.category_counts)} categories):")
        print("-" * 40)
        sorted_categories = sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            print(f"  {category}: {count}")
        print("-" * 40)
        print(f"  Total: {sum(self.category_counts.values())}")


def check_paths_in_list(paths: List[str]) -> Tuple[bool, List[str]]:
    """Check if all paths in the list exist"""
    missing_paths = []
    if not paths:
        return True, missing_paths
    
    for path in paths:
        if path and not os.path.exists(path):
            missing_paths.append(path)
    
    return len(missing_paths) == 0, missing_paths


def validate_data_format(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate if data format is compliant"""
    required_fields = ['messages', 'videos', 'audios']
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    if not isinstance(data.get('videos', []), list):
        return False, "videos field should be a list type"
    
    if not isinstance(data.get('audios', []), list):
        return False, "audios field should be a list type"
    
    if not isinstance(data.get('messages', []), list):
        return False, "messages field should be a list type"
    
    # Check quality_evaluation structure
    quality_eval = data.get('quality_evaluation', {})
    if not isinstance(quality_eval, dict):
        return False, "quality_evaluation field should be a dict type"
    
    scores = quality_eval.get('scores', {})
    if not isinstance(scores, dict):
        return False, "quality_evaluation.scores field should be a dict type"
    
    score_fields = ['answer_accuracy_score', 'question_logic_score', 'composite_score']
    for field in score_fields:
        if field in scores:
            value = scores[field]
            if not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    return False, f"{field} field should be a number type, current value: {value}"
    
    return True, ""


def get_score(data: Dict[str, Any], field: str, default: float = 0) -> float:
    """Safely get score field (from nested quality_evaluation.scores)"""
    quality_eval = data.get('quality_evaluation', {})
    scores = quality_eval.get('scores', {})
    value = scores.get(field, default)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def get_should_filter(data: Dict[str, Any]) -> bool:
    """Get should_filter field (from quality_evaluation)"""
    quality_eval = data.get('quality_evaluation', {})
    return quality_eval.get('should_filter', False)


def get_category_name(data: Dict[str, Any]) -> str:
    """Get category_name field (from question_classification)"""
    question_cls = data.get('question_classification', {})
    return question_cls.get('category_name', 'unknown')


def filter_data(data: Dict[str, Any], stats: FilterStats, check_paths: bool = True) -> Tuple[bool, str]:
    """Filter data according to rules"""
    if check_paths:
        # 1. Check video paths
        videos = data.get('videos', [])
        if videos:
            all_exist, missing = check_paths_in_list(videos)
            if not all_exist:
                stats.missing_video_path += 1
                return False, f"Video path not found: {missing}"
        
        # 2. Check audio paths
        audios = data.get('audios', [])
        if audios:
            all_exist, missing = check_paths_in_list(audios)
            if not all_exist:
                stats.missing_audio_path += 1
                return False, f"Audio path not found: {missing}"
    
    # 3. Check format
    is_valid, error_msg = validate_data_format(data)
    if not is_valid:
        stats.invalid_format += 1
        return False, f"Invalid format: {error_msg}"
    
    # 4. Check answer_accuracy_score == 10
    answer_accuracy = get_score(data, 'answer_accuracy_score', -1)
    if answer_accuracy != 10:
        stats.answer_accuracy_filtered += 1
        return False, f"answer_accuracy_score={answer_accuracy} (required=10)"
    
    # 5. Check question_logic_score >= 8
    question_logic = get_score(data, 'question_logic_score', 0)
    if question_logic < 8:
        stats.question_logic_filtered += 1
        return False, f"question_logic_score={question_logic} (required>=8)"
    
    # 6. Check composite_score >= 7
    composite = get_score(data, 'composite_score', 0)
    if composite < 7:
        stats.composite_score_filtered += 1
        return False, f"composite_score={composite} (required>=7)"
    
    # 7. Check should_filter is not true
    should_filter = get_should_filter(data)
    if should_filter in [True, 'true', 'True', 1, '1']:
        stats.should_filter_filtered += 1
        return False, "should_filter=true"
    
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Data quality filtering")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--skip_path_check", action="store_true", help="Skip path existence check")
    args = parser.parse_args()
    
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    
    stats = FilterStats()
    kept_data = []
    
    print(f"Starting to process file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print("-" * 60)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            stats.total += 1
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                stats.invalid_format += 1
                print(f"[Line {line_num}] JSON parsing error: {e}")
                continue
            
            keep, reason = filter_data(data, stats, check_paths=not args.skip_path_check)
            
            if keep:
                stats.kept += 1
                category = get_category_name(data)
                stats.category_counts[category] += 1
                kept_data.append(data)
            else:
                if "path not found" in reason.lower():
                    print(f"[Line {line_num}] Removed: {reason}")
            
            if stats.total % 10000 == 0:
                print(f"Processed {stats.total} entries, kept {stats.kept}...")
    
    print(f"\nSaving filtered data to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for data in kept_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    stats.print_summary()
    print(f"\nOutput file saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
