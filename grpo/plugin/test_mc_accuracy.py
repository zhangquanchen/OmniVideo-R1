#!/usr/bin/env python3
"""
Test script: Test the accuracy of option extraction in _compute_mc_accuracy function
"""

import re
from mathruler.grader import grade_answer


def compute_mc_accuracy(completion: str, solution: str) -> float:
    """Use mathruler.grader.grade_answer to evaluate the answer"""
    try:
        content_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if content_match:
            answer_text = content_match.group(1).strip()
            # Use grade_answer for evaluation
            is_correct = grade_answer(answer_text, solution)
            print(f"Student answer: {answer_text}")
            print(f"Ground truth: {solution}")
            print(f"Is correct: {is_correct}")
            return 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"Error: {e}")
    return 0.0

def compute_mc_accuracy_easy(completion: str, solution: str) -> float:
    try:
        # Extract answer from content between <answer> tags
        content_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if content_match:
            answer_text = content_match.group(1).strip()
            # Extract option letter (A, B, C, D, E, etc.)
            option_match = re.match(r'^([A-G])', answer_text)
            print(f"option_match: {option_match}")
            if option_match:
                student_option = option_match.group(1)
                print(f"student_option: {student_option}")
                # Extract option from solution
                sol_match = re.match(r'^([A-G])', solution.strip())
                if sol_match:
                    ground_truth_option = sol_match.group(1)
                    
                    if student_option == ground_truth_option:
                        return 1.0
    except Exception:
        pass
    return 0.0


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("<answer> Hirl </answer>", "A", "test")
    ]
    
    print("=" * 60)
    for completion, solution, desc in test_cases:
        result = compute_mc_accuracy_easy(completion, solution)
        print(f"Result: {result}")
        print("-" * 40)
