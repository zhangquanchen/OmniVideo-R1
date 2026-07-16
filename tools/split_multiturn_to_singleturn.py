#!/usr/bin/env python3
"""
Script to split multi-turn conversations into single-turn conversations
"""
import json
import sys
from pathlib import Path


def ensure_prefix(text, prefix="<video><audio>\n"):
    """Ensure text starts with specified prefix"""
    if not text.startswith(prefix):
        return prefix + text
    return text


def split_multiturn_conversation(data):
    """
    Split one multi-turn conversation data into multiple single-turn conversation data
    
    Args:
        data: Original data dictionary
        
    Returns:
        list: List of split single-turn conversation data
    """
    messages = data["messages"]
    num_turns = len(messages) // 2  # Each turn contains one user and one assistant message
    
    # If single-turn conversation, check and ensure correct prefix
    if num_turns == 1:
        # Ensure user content starts with <video><audio>\n
        user_content = messages[0]["content"]
        if not user_content.startswith("<video><audio>\n"):
            messages[0]["content"] = ensure_prefix(user_content)
        
        # Ensure problem starts with <video><audio>\n
        if "problem" in data:
            data["problem"] = ensure_prefix(data["problem"])
        
        return [data]
    
    # Multi-turn conversation, split into multiple single-turn conversations
    result = []
    for turn_idx in range(num_turns):
        # Extract current turn's user and assistant messages
        user_msg = messages[turn_idx * 2].copy()
        assistant_msg = messages[turn_idx * 2 + 1].copy()
        
        # Ensure user content starts with <video><audio>\n
        user_content = user_msg["content"]
        if not user_content.startswith("<video><audio>\n"):
            user_msg["content"] = ensure_prefix(user_content)
        
        # Extract question part from user content as problem
        problem = user_msg["content"]
        
        # Create new single-turn conversation data
        new_data = {
            "id": f"{data['id']}_turn{turn_idx + 1}",
            "Type": data["Type"],
            "messages": [user_msg, assistant_msg],
            "problem": problem,
            "solution": assistant_msg["content"],
            "videos": data["videos"],
            "audios": data["audios"],
            "data_source": data["data_source"]
        }
        
        result.append(new_data)
    
    return result


def print_demo(original_data, split_data_list):
    """Print conversion demo"""
    print("\n" + "=" * 80)
    print("Conversion Demo")
    print("=" * 80)
    
    print("\n[Original Data]")
    print(f"ID: {original_data['id']}")
    print(f"Turns: {len(original_data['messages']) // 2}")
    print(f"Messages count: {len(original_data['messages'])}")
    print(json.dumps(original_data, indent=2, ensure_ascii=False))
    
    print("\n" + "-" * 80)
    print(f"[After Split] (total {len(split_data_list)} entries)")
    print("-" * 80)
    
    for idx, split_data in enumerate(split_data_list, 1):
        print(f"\nEntry {idx}:")
        print(json.dumps(split_data, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80)


def verify_all_single_turn(output_file):
    """Verify all data in output file are single-turn conversations"""
    print(f"\nVerifying output file: {output_file}")
    print("Checking if all data are single-turn conversations...")
    
    multi_turn_found = []
    total_count = 0
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_count += 1
                num_turns = len(data['messages']) // 2
                
                if num_turns != 1:
                    multi_turn_found.append({
                        'line': line_num,
                        'id': data['id'],
                        'turns': num_turns
                    })
                    
            except Exception as e:
                print(f"Warning: Line {line_num} verification error: {e}")
    
    print(f"Verification complete! Total {total_count} entries checked")
    
    if multi_turn_found:
        print(f"\n⚠️  Found {len(multi_turn_found)} multi-turn conversation entries:")
        for item in multi_turn_found[:10]:  # Show only first 10
            print(f"  - Line {item['line']}, ID: {item['id']}, Turns: {item['turns']}")
        return False
    else:
        print("✓ All data are single-turn conversations!")
        return True


def process_jsonl_file(input_file, output_file):
    """
    Process JSONL file, split multi-turn conversations into single-turn conversations
    
    Args:
        input_file: Input file path
        output_file: Output file path
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_file}")
        sys.exit(1)
    
    print(f"Starting to process file: {input_file}")
    print(f"Output file: {output_file}")
    
    total_input = 0
    total_output = 0
    single_turn_count = 0
    multi_turn_count = 0
    demo_shown = False
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            if line_num % 1000 == 0:
                print(f"Processed {line_num} entries...")
            
            try:
                data = json.loads(line.strip())
                total_input += 1
                
                # Split conversation
                split_data_list = split_multiturn_conversation(data)
                
                # Show conversion demo for first multi-turn conversation
                if not demo_shown and len(split_data_list) > 1:
                    print_demo(data, split_data_list)
                    demo_shown = True
                
                # Statistics
                if len(split_data_list) == 1:
                    single_turn_count += 1
                else:
                    multi_turn_count += 1
                
                # Write to output file
                for split_data in split_data_list:
                    fout.write(json.dumps(split_data, ensure_ascii=False) + '\n')
                    total_output += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} JSON parsing error: {e}")
                continue
            except Exception as e:
                print(f"Warning: Line {line_num} processing error: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Input data total: {total_input}")
    print(f"  - Single-turn conversations: {single_turn_count}")
    print(f"  - Multi-turn conversations: {multi_turn_count} (already split)")
    print(f"Output data total: {total_output}")
    print(f"Output file: {output_file}")
    
    # Verify output file
    verify_all_single_turn(output_file)


def main():
    # Default file paths - please modify to your actual paths
    input_file = "./data/input.jsonl"
    output_file = "./data/output_singleturn.jsonl"
    
    # If command line arguments provided, use them
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    process_jsonl_file(input_file, output_file)


if __name__ == "__main__":
    main()
