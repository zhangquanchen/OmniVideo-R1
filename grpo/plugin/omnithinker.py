# existing reward functions
# ms-swift/swift/plugin/orm.py
# orms = {
#     'toolbench': ReactORM,
#     'math': MathORM,
#     'accuracy': MathAccuracy,
#     'format': Format,
#     'react_format': ReActFormat,
#     'cosine': CosineReward,
#     'repetition': RepetitionPenalty,
#     'soft_overlong': SoftOverlong,
# }

import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import torch

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""


# For additional reward functions, refer to swift/plugin/orm.py.
class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            # logger.debug(f"content: {content}, sol: {sol}")
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


orms['external_r1v_acc'] = MultiModalAccuracyORM

# Multi Choice Accuracy Reward Function
class MultiChoiceAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the multiple choice answer is correct.
        Extracts option letter from <answer> tags in completion and compares with solution.
        
        Args:
            completions (list[str]): Generated outputs (e.g., "<answer>E. tugs on her shirt.</answer>")
            solution (list[str]): Ground Truth answers (e.g., "A. ")

        Returns:
            list[float]: Reward scores (1.0 if correct, 0.0 otherwise)
        """
        rewards = []
        for content, sol in zip(completions, solution):
            # logger.debug("--------------------------------")
            # logger.debug(f"Content: {content}")
            # logger.debug(f"Solution: {sol}")
            print("--------------------------------")
            print(f"Content: {content}")
            print(f"Solution: {sol}")
            reward = 0.0
            try:
                # Extract answer from content between <answer> tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if content_match:
                    answer_text = content_match.group(1).strip()
                    # Extract option letter (A, B, C, D, E, etc.)
                    # Match pattern like "E. tugs on her shirt." -> "E"
                    option_match = re.match(r'^([A-Z])', answer_text)
                    if option_match:
                        student_option = option_match.group(1)
                        
                        # Extract option from solution (e.g., "A. " -> "A")
                        sol_match = re.match(r'^([A-Z])', sol.strip())
                        if sol_match:
                            ground_truth_option = sol_match.group(1)
                            
                            # Compare options
                            # logger.debug(f"Student Option: {student_option}, Ground Truth Option: {ground_truth_option}")
                            print(f"Student Option: {student_option}, Ground Truth Option: {ground_truth_option}")
                            if student_option == ground_truth_option:
                                reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if extraction fails
            print(f"Multi Choice Reward: {reward}")
            # logger.debug(f"Multi Choice Reward: {reward}")
            rewards.append(reward)
        return rewards


orms['mc_acc'] = MultiChoiceAccuracyORM


# Soft Multi Choice Accuracy Reward Function, using LLM to evaluate the correctness of answers.
class SoftMultiChoiceAccuracyORM(ORM):
    """
    Soft reward function that uses an LLM to evaluate the correctness of answers.
    Instead of exact matching, it calls a Qwen model to compare the extracted answer
    with the ground truth solution and returns a score.
    """

    def __init__(self, base_url="http://29.160.42.102:22002/v1", judge_model="Qwen3-VL-235B-Instruct", timeout=30):
        """
        Initialize the soft accuracy ORM with OpenAI-compatible API client.
        
        Args:
            base_url: API base URL for the Qwen model
            judge_model: Model name to use for evaluation (renamed to avoid conflict with --model arg)
            timeout: Request timeout in seconds
        """
        from openai import OpenAI
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=timeout
        )
        self.judge_model = judge_model
        self.evaluation_prompt = """You are an expert evaluator for multiple choice questions. Your task is to compare a student's answer with the ground truth answer and determine if they are semantically equivalent.

Ground Truth Answer: {ground_truth}

Student's Answer: {student_answer}

Please evaluate if the student's answer is correct by comparing it with the ground truth. Consider the following:
1. If both answers refer to the same option (e.g., both are "A" or both describe the same choice), they are correct.
2. Minor differences in wording or formatting should not affect correctness.
3. The semantic meaning is what matters, not the exact text.

Respond with ONLY a single number between 0.0 and 1.0:
- 1.0: The answer is completely correct
- 0.5: The answer is partially correct or ambiguous
- 0.0: The answer is incorrect

Your response (just the number):"""

    def _evaluate_with_llm(self, student_answer: str, ground_truth: str) -> float:
        """
        Use LLM to evaluate the correctness of the student's answer.
        
        Args:
            student_answer: The extracted answer from the completion
            ground_truth: The ground truth solution
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": self.evaluation_prompt.format(
                        ground_truth=ground_truth,
                        student_answer=student_answer
                    )
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=messages,
                max_tokens=16,
                temperature=0.0  # Use deterministic output for consistency
            )
            
            result_text = response.choices[0].message.content.strip()
            print(f"LLM Result Text: {result_text}")
            # Extract the score from the response
            # Try to parse as float directly
            try:
                score = float(result_text)
                # Clamp to [0.0, 1.0]
                score = max(0.0, min(1.0, score))
                return score
            except ValueError:
                # If parsing fails, try to find a number in the response
                import re
                match = re.search(r'([0-9]*\.?[0-9]+)', result_text)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                else:
                    logger.warning(f"Could not parse score from LLM response: {result_text}")
                    return 0.0
                    
        except Exception as e:
            logger.warning(f"Error calling LLM for evaluation: {e}")
            return 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Soft reward function that uses LLM to evaluate multiple choice answers.
        Extracts answer from <answer> tags and compares with solution using an LLM.
        
        Args:
            completions (list[str]): Generated outputs (e.g., "<answer>E. tugs on her shirt.</answer>")
            solution (list[str]): Ground Truth answers (e.g., "A. something")

        Returns:
            list[float]: Reward scores (0.0 to 1.0 based on LLM evaluation)
        """
        rewards = []
        for content, sol in zip(completions, solution):
            print("--------------------------------")
            print(f"Content: {content}")
            print(f"Solution: {sol}")
            reward = 0.0
            try:
                # Extract answer from content between <answer> tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if content_match:
                    student_answer = content_match.group(1).strip()
                    
                    # Use LLM to evaluate the answer
                    reward = self._evaluate_with_llm(student_answer, sol.strip())
                    print(f"Student Answer: {student_answer}\n")
                    print(f"Ground Truth: {sol.strip()}")
                else:
                    print("No <answer> tags found in completion")
                    reward = 0.0
                    
            except Exception as e:
                logger.warning(f"Error in SoftMultiChoiceAccuracyORM: {e}")
                reward = 0.0
                
            print(f"Soft Multi Choice Reward: {reward}")
            rewards.append(reward)
        return rewards


orms['soft_mc_acc'] = SoftMultiChoiceAccuracyORM


# Video Segment Caption Reward Function
class QueryIntentORM(ORM):
    """
    Reward function that evaluates video segment captions in two dimensions:
    1. Caption Accuracy (reward1): Whether each time segment's caption accurately describes the video content
    2. Segment Relevance (reward2): Whether the selected segments are relevant and sufficient for answering the question
    
    Expected input format:
    <time>1.0-2.5</time><caption>A person holds a paintbrush...</caption>
    <time>5.2-6.8</time><caption>The person paints a sun...</caption>
    <thinking>...</thinking>
    <answer>...</answer>
    """
    
    def __init__(self, base_url="http://29.160.42.102:22002/v1", judge_model="Qwen3-VL-235B-Instruct", 
                 timeout=120, temp_dir="/tmp/video_segments"):
        """
        Initialize the video segment caption ORM with OpenAI-compatible API client.
        
        Args:
            base_url: API base URL for the VLM model
            judge_model: Model name to use for evaluation
            timeout: Request timeout in seconds
            temp_dir: Directory to store temporary video segments
        """
        from openai import OpenAI
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=timeout
        )
        self.judge_model = judge_model
        self.temp_dir = temp_dir
        
        # Create temp directory if it doesn't exist
        import os
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Prompt for evaluating caption accuracy for a single video segment
        self.caption_accuracy_prompt = """You are an expert evaluator for video content description. Your task is to evaluate whether the provided caption accurately describes the video content.

Video segment is provided below.

Caption to evaluate: {caption}

Please evaluate the caption's accuracy based on the following criteria:
1. Does the caption correctly describe the main actions/events in the video?
2. Does the caption accurately describe the objects/people shown?
3. Does the caption capture the relevant audio information if present?
4. Is the caption neither too vague nor too specific compared to actual content?

Respond with ONLY a single number between 0.0 and 1.0:
- 1.0: The caption is completely accurate and comprehensive
- 0.7-0.9: The caption is mostly accurate with minor inaccuracies
- 0.4-0.6: The caption is partially accurate but misses important details
- 0.1-0.3: The caption has significant inaccuracies
- 0.0: The caption is completely incorrect or irrelevant

Your response (just the number):"""

        # Prompt for evaluating segment relevance to the question
        self.segment_relevance_prompt = """You are an expert evaluator for video-based question answering. Your task is to evaluate whether the selected video segments are relevant and sufficient for answering the given question.

The video segments shown are the key segments selected by the model to answer the question.

Question: {question}

Please evaluate the selected segments based on the following criteria:
1. Relevance: Do the segments contain information directly related to answering the question?
2. Sufficiency: Do the segments provide enough information to answer the question?
3. Precision: Are the segments focused without including unnecessary/redundant content?
4. Coverage: Do the segments cover all aspects needed to answer the question?

Respond with ONLY a single number between 0.0 and 1.0:
- 1.0: The segments are perfectly relevant, sufficient, and precise for answering the question
- 0.7-0.9: The segments are mostly relevant and sufficient with minor issues
- 0.4-0.6: The segments are partially relevant but may miss some key information or include redundancy
- 0.1-0.3: The segments have limited relevance to the question
- 0.0: The segments are completely irrelevant to the question

Your response (just the number):"""

    def _extract_time_caption_pairs(self, content: str) -> List[tuple]:
        """
        Extract (time_range, caption) pairs from model output.
        
        Args:
            content: Model output containing <time>...</time><caption>...</caption> patterns
            
        Returns:
            List of tuples: [((start_time, end_time), caption), ...]
        """
        pairs = []
        # Pattern to match <time>start-end</time><caption>...</caption>
        pattern = r'<time>\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*</time>\s*<caption>(.*?)</caption>'
        
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                start_time = float(match[0])
                end_time = float(match[1])
                caption = match[2].strip()
                pairs.append(((start_time, end_time), caption))
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing time-caption pair: {e}")
                continue
        
        return pairs

    def _extract_video_segment(self, video_path: str, start_time: float, end_time: float) -> str:
        """
        Extract a video segment and save to temporary file.
        
        Args:
            video_path: Path to the original video
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Path to the extracted video segment, or None if extraction fails
        """
        import subprocess
        import uuid
        import os
        
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None
        
        # Generate unique filename for the segment
        segment_filename = f"segment_{uuid.uuid4().hex[:8]}_{start_time:.1f}_{end_time:.1f}.mp4"
        segment_path = os.path.join(self.temp_dir, segment_filename)
        
        try:
            # Use ffmpeg to extract the video segment
            duration = end_time - start_time
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',  # Audio codec
                '-strict', 'experimental',
                '-loglevel', 'error',  # Suppress verbose output
                segment_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            
            if os.path.exists(segment_path):
                return segment_path
            else:
                logger.warning(f"Video segment extraction failed: output file not created")
                return None
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Video segment extraction timed out for {video_path}")
            return None
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg error extracting segment: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Error extracting video segment: {e}")
            return None

    def _concat_video_segments(self, segment_paths: List[str]) -> str:
        """
        Concatenate multiple video segments into one video.
        
        Args:
            segment_paths: List of paths to video segments
            
        Returns:
            Path to the concatenated video, or None if concatenation fails
        """
        import subprocess
        import uuid
        import os
        
        if not segment_paths:
            return None
        
        if len(segment_paths) == 1:
            return segment_paths[0]
        
        # Filter out None values
        valid_paths = [p for p in segment_paths if p and os.path.exists(p)]
        if not valid_paths:
            return None
        
        if len(valid_paths) == 1:
            return valid_paths[0]
        
        # Generate output filename
        concat_filename = f"concat_{uuid.uuid4().hex[:8]}.mp4"
        concat_path = os.path.join(self.temp_dir, concat_filename)
        
        # Create a file list for ffmpeg concat
        list_filename = f"concat_list_{uuid.uuid4().hex[:8]}.txt"
        list_path = os.path.join(self.temp_dir, list_filename)
        
        try:
            # Write the file list
            with open(list_path, 'w') as f:
                for segment_path in valid_paths:
                    f.write(f"file '{segment_path}'\n")
            
            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_path,
                '-c', 'copy',
                '-loglevel', 'error',
                concat_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            
            # Clean up the list file
            if os.path.exists(list_path):
                os.remove(list_path)
            
            if os.path.exists(concat_path):
                return concat_path
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error concatenating video segments: {e}")
            if os.path.exists(list_path):
                os.remove(list_path)
            return None

    def _encode_video_to_base64(self, video_path: str) -> str:
        """
        Encode video file to base64.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Base64 encoded string, or None if encoding fails
        """
        import base64
        import os
        
        if not video_path or not os.path.exists(video_path):
            return None
        
        try:
            with open(video_path, 'rb') as f:
                video_data = f.read()
            return base64.b64encode(video_data).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error encoding video to base64: {e}")
            return None

    def _evaluate_caption_accuracy(self, video_path: str, caption: str) -> float:
        """
        Use VLM to evaluate if the caption accurately describes the video segment.
        
        Args:
            video_path: Path to the video segment
            caption: Caption to evaluate
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            video_base64 = self._encode_video_to_base64(video_path)
            if not video_base64:
                logger.warning("Failed to encode video for caption evaluation")
                return 0.0
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": self.caption_accuracy_prompt.format(caption=caption)
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=messages,
                max_tokens=16,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            print(f"Caption Accuracy LLM Result: {result_text}")
            
            return self._parse_score(result_text)
            
        except Exception as e:
            logger.warning(f"Error evaluating caption accuracy: {e}")
            return 0.0

    def _evaluate_segments_relevance(self, video_path: str, question: str) -> float:
        """
        Use VLM to evaluate if the selected segments are relevant and sufficient for the question.
        
        Args:
            video_path: Path to the concatenated video of all segments
            question: The question to be answered
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            video_base64 = self._encode_video_to_base64(video_path)
            if not video_base64:
                logger.warning("Failed to encode video for relevance evaluation")
                return 0.0
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": self.segment_relevance_prompt.format(question=question)
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=messages,
                max_tokens=16,
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            print(f"Segment Relevance LLM Result: {result_text}")
            
            return self._parse_score(result_text)
            
        except Exception as e:
            logger.warning(f"Error evaluating segment relevance: {e}")
            return 0.0

    def _parse_score(self, result_text: str) -> float:
        """
        Parse score from LLM response.
        
        Args:
            result_text: Raw text response from LLM
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            score = float(result_text)
            return max(0.0, min(1.0, score))
        except ValueError:
            # Try to find a number in the response
            match = re.search(r'([0-9]*\.?[0-9]+)', result_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            else:
                logger.warning(f"Could not parse score from LLM response: {result_text}")
                return 0.0

    def _extract_question_from_problem(self, problem: str) -> str:
        """
        Extract the actual question from the problem text.
        
        Args:
            problem: Full problem text which may contain <video>, <audio> tags etc.
            
        Returns:
            Cleaned question text
        """
        # Remove video/audio tags
        question = problem.replace('<video>', '').replace('</video>', '')
        question = question.replace('<audio>', '').replace('</audio>', '')
        question = question.strip()
        return question

    def _cleanup_temp_files(self, file_paths: List[str]):
        """
        Clean up temporary video segment files.
        
        Args:
            file_paths: List of file paths to delete
        """
        import os
        for path in file_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Error removing temp file {path}: {e}")

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        """
        Compute rewards based on video segment caption accuracy and relevance.
        
        Args:
            completions: List of model completions
            solution: Ground truth solutions (not used in this reward)
            **kwargs: Additional data including 'videos' and 'problem'
            
        Returns:
            List of reward scores (average of caption accuracy and segment relevance)
        """
        rewards = []
        
        # Get videos and problems from kwargs
        videos_list = kwargs.get('videos', [])
        problems = kwargs.get('problem', [])
        
        for i, content in enumerate(completions):
            print("=" * 50)
            print(f"[QueryIntentORM] Processing sample {i}")
            
            # Get video path and problem for this sample
            video_path = None
            if i < len(videos_list) and videos_list[i]:
                video_data = videos_list[i]
                # Handle both list and string formats
                if isinstance(video_data, list) and len(video_data) > 0:
                    video_path = video_data[0]
                elif isinstance(video_data, str):
                    video_path = video_data
            
            question = ""
            if i < len(problems):
                question = self._extract_question_from_problem(problems[i]) if problems[i] else ""
            
            print(f"Video path: {video_path}")
            print(f"Question: {question[:200]}...")
            print(f"Content: {content[:300]}...")
            
            # Extract time-caption pairs from model output
            pairs = self._extract_time_caption_pairs(content)
            print(f"Found {len(pairs)} time-caption pairs")
            
            if not pairs or not video_path:
                print(f"No valid pairs or video path, assigning reward = 0.0")
                rewards.append(0.0)
                continue
            
            # Lists to track temp files for cleanup
            temp_files = []
            
            try:
                # ===== Reward 1: Caption Accuracy =====
                caption_scores = []
                segment_paths = []
                
                for (start_time, end_time), caption in pairs:
                    print(f"  Evaluating segment {start_time}-{end_time}: {caption[:50]}...")
                    
                    # Extract video segment
                    segment_path = self._extract_video_segment(video_path, start_time, end_time)
                    
                    if segment_path:
                        temp_files.append(segment_path)
                        segment_paths.append(segment_path)
                        
                        # Evaluate caption accuracy for this segment
                        score = self._evaluate_caption_accuracy(segment_path, caption)
                        caption_scores.append(score)
                        print(f"  Caption accuracy score: {score}")
                    else:
                        print(f"  Failed to extract segment, skipping")
                
                # Calculate average caption accuracy (reward1)
                reward1 = sum(caption_scores) / len(caption_scores) if caption_scores else 0.0
                print(f"Reward1 (Caption Accuracy): {reward1}")
                
                # ===== Reward 2: Segment Relevance =====
                reward2 = 0.0
                if segment_paths and question:
                    # Concatenate all segments
                    concat_path = self._concat_video_segments(segment_paths)
                    
                    if concat_path:
                        if concat_path not in temp_files:
                            temp_files.append(concat_path)
                        
                        # Evaluate segment relevance
                        reward2 = self._evaluate_segments_relevance(concat_path, question)
                        print(f"Reward2 (Segment Relevance): {reward2}")
                    else:
                        print(f"Failed to concatenate segments, reward2 = 0.0")
                else:
                    print(f"No segments or question available, reward2 = 0.0")
                
                # ===== Final Reward =====
                final_reward = (reward1 + reward2) / 2.0
                print(f"Final Reward: {final_reward}")
                rewards.append(final_reward)
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                rewards.append(0.0)
                
            finally:
                # Cleanup temporary files
                self._cleanup_temp_files(temp_files)
        
        return rewards


orms['query_intent'] = QueryIntentORM


# Think Answer Format Reward Function
class ThinkAnswerFormatORM(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion follows the format:
        <thinking> thinking process </thinking> <answer> final answer </answer>
        
        This rewards outputs that follow the instruction:
        "Please respond with only the letter of the correct answer.
        First output the thinking process in <thinking> </thinking> tags 
        and then output the final answer (option) in <answer> </answer> tags."
        
        Args:
            completions (list[str]): Generated outputs

        Returns:
            list[float]: Reward scores (1.0 if format is correct, 0.0 otherwise)
        """
        rewards = []
        for content in completions:
            reward = 0.0
            try:
                # Check if the format contains: <thinking>...</thinking> <answer>...</answer>
                # Allows other content before and after
                pattern = r'<thinking>.*?</thinking>\s*<answer>.*?</answer>'
                
                # Check if the pattern exists in content
                if re.search(pattern, content, re.DOTALL):
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if check fails
            # logger.debug("--------------------------------")
            # logger.debug(f"Format Reward: {reward}")
            print("--------------------------------")
            print(f"Format Reward: {reward}")
            rewards.append(reward)
        return rewards


orms['think_answer_format'] = ThinkAnswerFormatORM


# Time Caption Think Answer Format Reward Function
class TimeCaptionThinkAnswerFormatORM(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion follows the format:
        Multiple <time>start-end</time><caption>description</caption> segments
        followed by <thinking>...</thinking><answer>...</answer>
        
        Expected format example:
        <time>1.0-2.5</time><caption>A person holds a paintbrush...</caption>
        <time>5.2-6.8</time><caption>The person paints a sun...</caption>
        <thinking>Let me think...</thinking>
        <answer>Painting a sun on a canvas</answer>
        
        Note: 
        - Every <time> must have a corresponding <caption> after it.
        - Every <caption> must have a corresponding <time> before it.
        - The number of <time> tags must equal the number of <caption> tags.
        
        Args:
            completions (list[str]): Generated outputs

        Returns:
            list[float]: Reward scores (1.0 if format is correct, 0.0 otherwise)
        """
        rewards = []
        for content in completions:
            reward = 0.0
            num_time_caption_pairs = 0
            num_total_times = 0
            num_total_captions = 0
            has_think_answer = False
            try:
                # Pattern for <time>start-end</time><caption>...</caption> pairs
                # Time format: decimal numbers like 1.0-2.5
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
                
                # Only give reward if:
                # 1. At least one time-caption pair exists
                # 2. All times and captions are properly paired (no standalone time or caption)
                # 3. thinking-answer format exists
                if num_time_caption_pairs >= 1 and all_properly_paired and has_think_answer:
                    reward = 1.0
                    
            except Exception:
                pass  # Keep reward as 0.0 if check fails
            
            print("--------------------------------")
            print(f"TimeCaptionThinkAnswer Format Reward: {reward}")
            rewards.append(reward)
        return rewards


orms['tcta_format'] = TimeCaptionThinkAnswerFormatORM


# ==================== Multi-Modal ModalityAttention Comparison Reward ====================
# Global registry for storing modality_attention rollout results
class MultiModalModalityAttentionComparisonORM(ORM):
    """
    Reward function that compares accuracy across original, no-video, and no-audio variants.
    
    This ORM requires the trainer to generate completions for all three variants and store
    them in the batch data with keys:
    - 'modality_attention_no_video_completion': completion without video
    - 'modality_attention_no_audio_completion': completion without audio
    
    The reward is 0.3 if original accuracy >= both modality_attention accuracies, 0.0 otherwise.
    """
    
    def __init__(self, comparison_reward: float = 0.3):
        """
        Args:
            comparison_reward: Reward value when original >= both modality_attention variants (default 0.3)
        """
        self.comparison_reward = comparison_reward

    def _compute_mc_accuracy(self, completion: str, solution: str) -> float:
        """
        Compute multiple choice accuracy for a single completion.
        
        Args:
            completion: Model's completion text
            solution: Ground truth solution
            
        Returns:
            1.0 if correct, 0.0 otherwise
        """
        try:
            # Extract answer from content between <answer> tags
            content_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
            if content_match:
                answer_text = content_match.group(1).strip()
                # Extract option letter (A, B, C, D, E, etc.)
                option_match = re.match(r'^([A-Z])', answer_text)
                if option_match:
                    student_option = option_match.group(1)
                    
                    # Extract option from solution
                    sol_match = re.match(r'^([A-Z])', solution.strip())
                    if sol_match:
                        ground_truth_option = sol_match.group(1)
                        
                        if student_option == ground_truth_option:
                            return 1.0
        except Exception:
            pass
        return 0.0
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Compute comparison-based rewards.
        
        Args:
            completions: Original completions (with video+audio)
            solution: Ground truth solutions
            **kwargs: Must contain 'modality_attention_no_video_completion' and 'modality_attention_no_audio_completion'
            
        Returns:
            List of rewards (0.3 if original >= both modality_attention variants, 0.0 otherwise)
        """
        # Get modality_attention completions from kwargs
        # print("--------------------------------")
        # print(f"kwargs: {kwargs}")
        no_video_completions = kwargs.get('modality_attention_no_video_completion', [])
        no_audio_completions = kwargs.get('modality_attention_no_audio_completion', [])
        
        if not no_video_completions or not no_audio_completions:
            logger.warning("ModalityAttention completions not found in kwargs. "
                          "Make sure trainer is configured to generate modality_attention variants.")
            # Return 0 reward if modality_attention data is missing
            return [0.0] * len(completions)
        
        rewards = []
        for i, (orig_comp, sol) in enumerate(zip(completions, solution)):
            # Compute accuracy for each variant
            orig_acc = self._compute_mc_accuracy(orig_comp, sol)
            
            no_video_comp = no_video_completions[i] if i < len(no_video_completions) else ""
            no_audio_comp = no_audio_completions[i] if i < len(no_audio_completions) else ""
            
            no_video_acc = self._compute_mc_accuracy(no_video_comp, sol)
            no_audio_acc = self._compute_mc_accuracy(no_audio_comp, sol)
            
            # Compare: reward if original >= both modality_attention variants
            if orig_acc >= no_video_acc and orig_acc >= no_audio_acc:
                reward = self.comparison_reward
            else:
                reward = 0.0
            
            print("=" * 40)
            print(f"[MultiModal ModalityAttention Comparison] Sample {i}")
            print(f"  Original completion: {orig_comp[:200]}...")
            print(f"  No-video completion: {no_video_comp[:200] if no_video_comp else 'N/A'}...")
            print(f"  No-audio completion: {no_audio_comp[:200] if no_audio_comp else 'N/A'}...")
            print(f"  Accuracies - Original: {orig_acc}, No-Video: {no_video_acc}, No-Audio: {no_audio_acc}")
            print(f"  ModalityAttention Comparison Reward: {reward}")
            
            rewards.append(reward)
        
        return rewards


orms['modality_attention'] = MultiModalModalityAttentionComparisonORM


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


orms['external_tooluse_length_reward'] = ToolUseLengthReward

"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs, **kwargs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step (Required): Constructs the next round of the infer request.
            - check_finished (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
            or override run method in MultiTurnScheduler class.

        Both methods accept:
            - the last turn's InferRequest/response_choice
            - the current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        swift rollout \
            --external_plugins /path/to/plugin.py \
            --multi_turn_scheduler my_scheduler
"""


class ToolCallScheduler(MultiTurnScheduler):
    # A simple scheduler that supports tool calls by overriding the `step` method
    # Tool parsing uses the ReAct format
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A simple tool registry. Extend or replace with your own tools as needed.
        self.tools = {
            'calculator': self._calculator_tool,
        }

    def _calculator_tool(self, expression: str) -> str:
        # A very small sandboxed calculator
        # The calculator tool implemented here can perform only basic arithmetic operations and
        # may not be able to solve all math problems in the dataset.
        import ast
        import operator

        def _evaluate_ast_node(node) -> Union[int, float]:
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f'Unsupported constant type: {type(node.value)}')

            elif isinstance(node, ast.Num):
                return node.n

            elif isinstance(node, ast.BinOp):
                left = _evaluate_ast_node(node.left)
                right = _evaluate_ast_node(node.right)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported operation: {type(node.op).__name__}')

                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError('Division by zero')

                return op(left, right)

            elif isinstance(node, ast.UnaryOp):
                operand = _evaluate_ast_node(node.operand)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported unary operation: {type(node.op).__name__}')

                return op(operand)

            else:
                raise TypeError(f'Unsupported AST node type: {type(node).__name__}')

        try:
            expression = expression.strip().replace(' ', '')

            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return 'Error: expression contains disallowed characters.'

            if expression.count('(') != expression.count(')'):
                return 'Error: unmatched parentheses.'

            try:
                result = ast.literal_eval(expression)
                return f'Result: {result}'
            except (ValueError, SyntaxError):
                node = ast.parse(expression, mode='eval')
                result = _evaluate_ast_node(node.body)
                return f'Result: {result}'

        except Exception as e:
            return f'Calculation error: {e}'

    def _extract_tool_calls(self, text: str):
        """
        Parse tool-call patterns using ReAct format from model output.
        Format: Action: tool_name\nAction Input: parameters
        """
        import re

        pattern = r'Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return [{'tool': name.strip(), 'params': params.strip()} for name, params in matches]

    def _execute_tools(self, tool_calls):
        """Run each requested tool and collect its observation string."""
        results = []
        for call in tool_calls:
            name, params = call['tool'], call['params']
            if name in self.tools:
                try:
                    result = self.tools[name](params)
                    results.append(result)
                except Exception as e:
                    results.append(f'tool error {e}')
            else:
                results.append(f'unknown tool {name}')
        return results

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        completion = response_choice.message.content
        tool_calls = self._extract_tool_calls(completion)
        if tool_calls is None:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        tool_calls = self._extract_tool_calls(completion)
        # assert len(tool_calls) == 1, 'this scheduler is designed for one tool call per turn'
        tool_results = self._execute_tools(tool_calls)
        # append tool result to the completion
        infer_request.messages[-1]['content'] += (tool_results[0])

        tokenizer = self.infer_engine.default_template.tokenizer
        result_tokens = tokenizer.encode(tool_results[0], add_special_tokens=False)
        token_ids.extend(result_tokens)
        loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_results': tool_results[0],
                'num_turns': current_turn,
            }
        }


multi_turns['tool_call_scheduler'] = ToolCallScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager
