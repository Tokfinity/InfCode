import json
import re
import os
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable
from datasets import load_dataset
import random
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

def failure_analysis_prompt(problem_statement: str, hints_text: str, working_trajectory: str, failed_trajectory: str, code_agent_code: str) -> str:
    """
    Generate a failure analysis prompt for comparing successful and failed trajectories.
    
    Creates a structured prompt that asks an AI model to analyze why one trajectory
    succeeded while another failed, and what the failed trajectory could have done
    differently to succeed.
    """
    return f"""Here's a software engineering issue: {problem_statement}.

Here's the hints: {hints_text}.

Here's the trajectory that fixed the issue: {working_trajectory}.

Here's the trajectory that failed: {failed_trajectory}.

Here's the prompts/tools code for the software engineering agent: {code_agent_code}.

Why does the first SWE-agent resolve the issue but the second failed?

What could the second agent have done differently to resolve the issue?

What improvements could be made to the prompt/tool usage?

Answer in the following format:
# Why the second agent failed 
...
# What could the second agent have done differently
...
# What improvements could be made to the prompt/tool usage
..."""

def openrouter_gemini_client(prompt: str, max_retries: int = 10, base_delay: float = 1.0) -> str:
    """
    Send a prompt to OpenRouter API using Gemini model for analysis.
    
    Makes API calls to OpenRouter service using Google's Gemini 2.5 Pro model
    with exponential backoff retry logic. Handles authentication, rate limiting,
    and error recovery for robust API communication.
    
    Args:
        prompt (str): The analysis prompt to send to the model
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 10.
        base_delay (float, optional): Base delay in seconds for exponential backoff. Defaults to 1.0.
        
    Returns:
        str: The model's response content as a string
        
    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set
        Exception: If API request fails after all retry attempts or returns invalid response
        
    Note:
        Requires OPENROUTER_API_KEY environment variable to be set with valid API key.
        Uses exponential backoff: 1s, 2s, 4s, 8s, 16s, etc. between retries.
    """
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not found")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "google/gemini-2.5-pro",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 200000,  # Increased limit for comprehensive analysis
        "temperature": 0.7  # Lower temperature for more consistent summaries
    }
    
    for attempt in range(max_retries):
        try:
            print(f"ATTEMPT: API attempt {attempt + 1}/{max_retries}...")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
                        
            if 'choices' not in result or not result['choices']:
                raise Exception(f"Invalid API response: missing or empty choices")
            
            content = result['choices'][0]['message']['content']
            
            if not content or content.strip() == "":
                print(f"WARNING: Empty content returned from API on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) / (2 ** random.uniform(0, 1))  # Exponential backoff with jitter
                    print(f"WAIT: Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Empty content returned from API after {max_retries} attempts")
            
            print(f"SUCCESS: API request successful on attempt {attempt + 1}")
            return content.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"FAILED: Request failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) / (2 ** random.uniform(0, 1))  # Exponential backoff with jitter
                print(f"WAIT: Waiting {delay:.2f} seconds before retry...")
                time.sleep(delay)
                continue
            else:
                raise Exception(f"FAILED: OpenRouter API request failed after {max_retries} attempts: {e}")
        
        except (KeyError, IndexError) as e:
            raise Exception(f"FAILED: Unexpected response format from OpenRouter API: {e}")
    
    raise Exception(f"FAILED: All {max_retries} attempts failed")

def get_swe_bench_problem_and_hints_for_split(instance_id: str, split_name: str) -> tuple[str, str]:
    """
    Retrieve problem statement and hints for a specific SWE-Bench instance in a specific split.
    
    Loads the appropriate SWE-Bench dataset and searches for the specified instance ID to extract the problem statement and hints text. 
    This data is used for generating analysis prompts and understanding the context of the software engineering problem.
    
    Args:
        instance_id (str): The unique identifier for the SWE-Bench instance
                          (e.g., 'scikit-learn__scikit-learn-10908')
        split_name (str): The split name (e.g., "Verified", "Lite", "Test", "Multimodal", "bash-only")
        
    Returns:
        tuple[str, str]: A tuple containing (problem_statement, hints_text)
        
    Raises:
        ValueError: If the specified instance_id is not found in the dataset
    Note:
        This function loads the full SWE-Bench dataset for the specified split, which may take some time on first execution due to dataset download and caching.
    """
    dataset_mapping = {
        "Verified": "princeton-nlp/SWE-bench_Verified",
        "Lite": "SWE-bench/SWE-bench_Lite", 
        "Test": "SWE-bench/SWE-bench",
        "Multimodal": "SWE-bench/SWE-bench_Multimodal",
        # Note: bash-only dataset not yet available on Hugging Face
    }
    
    dataset_name = dataset_mapping.get(split_name)
    if not dataset_name:
        raise ValueError(f"No dataset mapping found for split '{split_name}'")
    
    ds = load_dataset(dataset_name, split="test")
    target_row = None
    for row in ds:
        if row['instance_id'] == instance_id:
            target_row = row
            break
    if target_row is None:
        raise ValueError(f"Instance ID '{instance_id}' not found in the {split_name} dataset")
    problem_statement = target_row['problem_statement']
    hints_text = target_row['hints_text']
    return problem_statement, hints_text