import json
import argparse


def add_data_source_and_merge(llava_file, videovista_file, output_file):
    """
    Merge two JSONL files, add data_source field for VideoVista data
    
    Args:
        llava_file: LLaVA data file path
        videovista_file: VideoVista data file path
        output_file: Merged output file path
    """
    merged_data = []
    
    # Read llava file (already has data_source field)
    print(f"Reading llava file: {llava_file}")
    with open(llava_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                merged_data.append(data)
    print(f"llava data count: {len(merged_data)}")
    
    # Read VideoVista file and add data_source field
    print(f"Reading VideoVista file: {videovista_file}")
    videovista_count = 0
    with open(videovista_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                data['data_source'] = 'videovista'
                merged_data.append(data)
                videovista_count += 1
    print(f"VideoVista data count: {videovista_count}")
    
    # Write merged file
    print(f"Writing merged file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in merged_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Merge complete! Total data count: {len(merged_data)}")


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL files")
    parser.add_argument("--llava", "-l", required=True, help="LLaVA data file path")
    parser.add_argument("--videovista", "-v", required=True, help="VideoVista data file path")
    parser.add_argument("--output", "-o", required=True, help="Merged output file path")
    args = parser.parse_args()
    
    add_data_source_and_merge(args.llava, args.videovista, args.output)


if __name__ == "__main__":
    main()
