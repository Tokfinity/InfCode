import json
import os
import yaml
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from trajectory_discovery import find_failed_trajectory, find_successful_trajectory_for_split, download_top_agents_logs_for_split, count_top_agents_solved_for_split
from pairwise_analysis import get_swe_bench_problem_and_hints_for_split, failure_analysis_prompt, openrouter_gemini_client


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_code_agent_code(prompts_path: str, tools_path: str) -> Dict[str, str]:
    """
    Extract content from Code-agent directory including prompts_manager.py and tools/ files.
    """
    result = ""
    prompts_path = Path(prompts_path)
    if prompts_path.exists():
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                result += f"{prompts_path.name}: \n" + "```python\n" + f.read() + "```\n"
        except Exception as e:
            print(f"Error reading prompts_manager.py: {e}")   
    tools_path = Path(tools_path)
    if tools_path.exists():
        for file_path in tools_path.iterdir():
            if file_path.is_file() and not file_path.name.startswith('test'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result += f"{file_path.name}: \n" + "```python\n" + f.read() + "```\n"
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return result

def analyze_failure_causes(instance_id: str, current_directory: str, code_agent_code: str, split_name: str, num_agents: int = 5) -> Optional[str]:
    """
    Compare trajectories alongside test results to analyze specific failure causes.
    
    This function finds both failed and successful trajectories for a given instance, retrieves the problem context, and uses an LLM to analyze why the failure occurred.
    
    Args:
        instance_id: The instance ID to analyze
        current_directory: Directory containing failed trajectories
        code_agent_code: Code agent code for analysis
        split_name: The split name (e.g., "Verified", "Lite", "Test", "Multimodal", "bash-only")
        num_agents: Number of top agents to check for successful trajectories
    """
    failed_trajectory_path = find_failed_trajectory(instance_id, current_directory)
    if not failed_trajectory_path:
        print(f"No failed trajectory found for {instance_id}")
        return None    
    successful_trajectory_path = find_successful_trajectory_for_split(instance_id, split_name, num_agents)
    if not successful_trajectory_path:
        print(f"No successful trajectory found for {instance_id} in {split_name} split")
        return None
    try:
        with open(failed_trajectory_path, 'r', encoding='utf-8') as f:
            failed_content = f.read()
        with open(successful_trajectory_path, 'r', encoding='utf-8') as f:
            successful_content = f.read()        
        problem_statement, hints_text = get_swe_bench_problem_and_hints_for_split(instance_id, split_name)        
        prompt = failure_analysis_prompt(problem_statement, hints_text, successful_content, failed_content, code_agent_code)        
        analysis_result = openrouter_gemini_client(prompt)        
        formatted_result = f"Instance: {instance_id}\nAnalysis: {analysis_result}\n"
        print(f"Analysis result for {instance_id}: {analysis_result}")
        return formatted_result
    except Exception as e:
        print(f"Error analyzing {instance_id}: {e}")
        return None

def batch_failure_analysis(unresolved_id_txt_path: str, intermediate_output_path: str, output_path: str, logs_directory: str, prompts_path: str, tools_path: str, split_name: str, max_workers: int = 20, num_agents: int = 5) -> None:
    """
    Complete batch failure analysis pipeline for a specific split.    
    Process failed trajectories, find successful trajectories, analyze failure causes, and categorize them by cause. Generate a comprehensive report with recommendations and save it to output_path.
    
    Args:
        unresolved_id_txt_path: Path to text file containing unresolved instance IDs
        intermediate_output_path: Path to directory for intermediate output files
        output_path: Path to final output report file
        logs_directory: Directory containing failed trajectory logs
        prompts_path: Path to prompts manager Python file
        tools_path: Path to tools directory
        split_name: The split name (e.g., "Verified", "Lite", "Test", "Multimodal", "bash-only")
        max_workers: Maximum number of worker threads
        num_agents: Number of top agents to analyze
    """    
    print(f"Regenerating top agents performance CSV for {split_name} split...")
    count_top_agents_solved_for_split(split_name, num_agents)
    print(f"Ensuring experiments repository and logs are up to date for {split_name} split...")
    download_top_agents_logs_for_split("experiments", split_name, num_agents)
    try:
        with open(unresolved_id_txt_path, 'r', encoding='utf-8') as f:
            unresolved_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        if not unresolved_ids:
            print("No unresolved instances found in text file")
            return
        print(f"Found {len(unresolved_ids)} unresolved instances to analyze")
    except Exception as e:
        print(f"Error loading text file: {e}")
        return
    
    failure_analysis_results = []
    skipped_instances = []
    code_agent_code = extract_code_agent_code(prompts_path, tools_path)
    # create the intermediate output directory if it doesn't exist
    os.makedirs(intermediate_output_path, exist_ok=True)
    
    def process_instance(instance_id):
        print(f"\nProcessing: {instance_id}")
        try:
            analysis_result = analyze_failure_causes(instance_id, logs_directory, code_agent_code, split_name, num_agents)
            if analysis_result:
                with open(os.path.join(intermediate_output_path, f'{instance_id}.txt'), 'w', encoding='utf-8') as f:
                    f.write(analysis_result)
                return (instance_id, analysis_result, None)
            else:
                print(f"Skipped {instance_id} - no trajectories found")
                return (instance_id, None, "no trajectories found")
        except Exception as e:
            print(f"Error processing {instance_id}: {e}")
            return (instance_id, None, str(e))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_instance, instance_id) for instance_id in unresolved_ids]
        for future in as_completed(futures):
            instance_id, analysis_result, error = future.result()
            if analysis_result:
                failure_analysis_results.append(analysis_result)
            else:
                skipped_instances.append(instance_id)
    # combine the intermediate output files into a single file
    with open(output_path, 'w', encoding='utf-8') as f:
        for file in os.listdir(intermediate_output_path):
            if file.endswith('.txt'):
                with open(os.path.join(intermediate_output_path, file), 'r', encoding='utf-8') as f1:
                    f.write(f1.read() + '\n')
    # read the combined failure analysis report
    with open(output_path, 'r', encoding='utf-8') as f:
        failure_analysis_report = f.read()
    batch_prompt = f"""Read failure analysis reports and group failure causes into categories.

Here are the failure analysis reports:

{failure_analysis_report}

Here are the prompts/tools code for the software engineering agent:

{code_agent_code}

Please read the above analysis and categorize the failure causes into categories.

For each category, please describe the failure cause in detail, list related instances, and suggest improvements for the prompt/tool usage to address the failure cause. 

Format your response as:

# Problem 1: 
**Description**: [Your analysis of this problem]
**Related Instances**: 
- instance_id_1: [brief description of the failure]
- instance_id_2: [brief description of the failure]
**Improvements**: [Suggestions for improving the prompt/tool usage to address the failure cause]

# Problem 2:
**Description**: [Your analysis of this problem]
**Related Instances**: 
- instance_id_3: [brief description of the failure]
- instance_id_4: [brief description of the failure]
**Improvements**: [Suggestions for improving the prompt/tool usage to address the failure cause]

Continue for all identified problems."""
    
    try:
        batch_analysis = openrouter_gemini_client(batch_prompt)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(batch_analysis)
    except Exception as e:
        print(f"Error generating batch analysis: {e}")
        return f"Error generating batch analysis: {e}"

if __name__ == "__main__":
    config = load_config("config.yaml")
    
    # Validate paths exist
    paths = config["paths"]
    for path_name in ["txt_path", "logs_directory", "prompts_path", "tools_path"]:
        if not os.path.exists(paths[path_name]):
            raise FileNotFoundError(f"Path {path_name}: {paths[path_name]} does not exist")
    
    # Validate dataset split
    split_name = config["dataset"]["split"]
    valid_splits = ["Verified", "Lite", "Test", "Multimodal"]  # bash-only not yet available on Hugging Face
    if split_name not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got: {split_name}")
    
    # Validate max_workers
    max_workers = config["settings"]["max_workers"]
    if not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError(f"max_workers must be positive integer, got: {max_workers}")
    
    # Validate num_agents
    num_agents = config["settings"]["num_agents"]
    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValueError(f"num_agents must be positive integer, got: {num_agents}")
    
    batch_failure_analysis(
        paths["txt_path"], 
        paths["intermediate_output_path"], 
        paths["output_path"], 
        paths["logs_directory"], 
        paths["prompts_path"], 
        paths["tools_path"], 
        split_name,
        max_workers,
        num_agents
    )
