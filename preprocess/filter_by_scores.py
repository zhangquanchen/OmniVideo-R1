import json
import os
import argparse

def filter_and_remove_fields(input_path, output_high_path=None, output_rest_path=None, 
                             fields_to_remove=None, min_score=8):
    """
    Read JSONL file, filter data by video_dependency_score and audio_dependency_score,
    remove specified fields, and save to two separate files
    
    Args:
        input_path: Input file path
        output_high_path: Output file path for high score data
        output_rest_path: Output file path for remaining data
        fields_to_remove: List of fields to remove
        min_score: Minimum score threshold (both video and audio must be >= this value)
    """
    if fields_to_remove is None:
        fields_to_remove = ['quality_evaluation', 'question_classification']
    
    if output_high_path is None:
        base, ext = os.path.splitext(input_path)
        output_high_path = f"{base}_high_score{ext}"
    
    if output_rest_path is None:
        base, ext = os.path.splitext(input_path)
        output_rest_path = f"{base}_rest{ext}"
    
    high_count = 0
    rest_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_high_path, 'w', encoding='utf-8') as f_high, \
         open(output_rest_path, 'w', encoding='utf-8') as f_rest:
        
        for line in f_in:
            data = json.loads(line.strip())
            total_count += 1
            
            # Get scores
            scores = data.get('quality_evaluation', {}).get('scores', {})
            video_score = scores.get('video_dependency_score', 0)
            audio_score = scores.get('audio_dependency_score', 0)
            
            # Check if condition is met: both scores >= min_score
            is_high_score = video_score >= min_score and audio_score >= min_score
            
            # Remove specified fields
            for field in fields_to_remove:
                data.pop(field, None)
            
            # Write to corresponding file
            if is_high_score:
                f_high.write(json.dumps(data, ensure_ascii=False) + '\n')
                high_count += 1
            else:
                f_rest.write(json.dumps(data, ensure_ascii=False) + '\n')
                rest_count += 1
            
            if total_count % 10000 == 0:
                print(f"Processed {total_count} entries...")
    
    print(f"\nDone! Processed {total_count} entries in total")
    print(f"=" * 50)
    print(f"Data meeting condition (video_dependency_score >= {min_score} and audio_dependency_score >= {min_score}): {high_count} entries")
    print(f"Remaining data: {rest_count} entries")
    print(f"=" * 50)
    print(f"High score data output file: {output_high_path}")
    print(f"Remaining data output file: {output_rest_path}")
    
    return output_high_path, output_rest_path, high_count, rest_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter data by scores")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--min_score", "-s", type=int, default=7, help="Minimum score threshold")
    parser.add_argument("--output_high", "-oh", help="High score data output file path")
    parser.add_argument("--output_rest", "-or", help="Remaining data output file path")
    args = parser.parse_args()
    
    filter_and_remove_fields(
        args.input, 
        output_high_path=args.output_high,
        output_rest_path=args.output_rest,
        min_score=args.min_score
    )
