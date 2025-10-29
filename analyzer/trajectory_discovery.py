from __future__ import annotations
import os
import re
import subprocess
import sys
import json
import pathlib
import csv
import urllib.request
import argparse
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from utils import read_json_url_or_path
from datasets import load_dataset
from utils import find_failed_trajectory_in_issues_directory

LEADERBOARD_JSON = "https://raw.githubusercontent.com/swe-bench/swe-bench.github.io/master/data/leaderboards.json"
EXPERIMENTS_DIR = str(Path(__file__).parent / "experiments")
EXPERIMENTS_REPO_URL = "https://github.com/SWE-bench/experiments.git"

def top_agents_for_split(leaderboards_json: dict, split_name: str, num_agents: int = 5) -> list[dict]:
    """
    Get the top N SWE-bench agents for a specific split from the leaderboard.
    
    Args:
        leaderboards_json: The leaderboard JSON data
        split_name: The split name (e.g., "Verified", "Lite", "Test", "Multimodal")
        num_agents: Number of top agents to return (default: 5)
    
    Returns:
        List of top N agents sorted by resolution rate
    """
    lbs = leaderboards_json.get("leaderboards") or leaderboards_json
    split_lb = next((lb for lb in lbs if (lb.get("name") or "").lower() == split_name.lower()), None)
    if not split_lb:
        raise KeyError(f"Could not find '{split_name}' leaderboard in leaderboards.json")
    rows = [r for r in split_lb.get("results", []) if not r.get("warning")]
    rows.sort(key=lambda r: float(r.get("resolved", 0)), reverse=True)
    return rows[:num_agents]

def read_submission_resolved_ids(submission_dir: pathlib.Path) -> set[str]:
    """
    Read the resolved instance_ids from the results.json file in a SWE-bench verified submission folder.
    """
    fp = submission_dir / "results" / "results.json"
    if not fp.is_file():
        return set()
    try:
        j = json.loads(fp.read_text(encoding="utf-8"))
        return set(j.get("resolved", []))
    except Exception:
        return set()

def load_canonical_ids_for_split(split_name: str) -> list[str]:
    """
    Load canonical instance IDs for a specific SWE-bench split.
    
    Args:
        split_name: The split name (e.g., "Verified", "Lite", "Test", "Multimodal")
    
    Returns:
        List of sorted instance IDs for the split
    """
    dataset_mapping = {
        "Verified": "princeton-nlp/SWE-bench_Verified",
        "Lite": "SWE-bench/SWE-bench_Lite", 
        "Test": "SWE-bench/SWE-bench",
        "Multimodal": "SWE-bench/SWE-bench_Multimodal",
        # Note: bash-only dataset not available on Hugging Face
    }
    
    dataset_name = dataset_mapping.get(split_name)
    if not dataset_name:
        print(f"Warning: No dataset mapping found for split '{split_name}'")
        return []
    
    try:
        ds = load_dataset(dataset_name, split="test")
        ids = sorted(ds["instance_id"])
        print(f"Loaded {len(ids)} instance IDs for {split_name} split")
        return ids
    except Exception as e:
        print(f"Error loading dataset for {split_name}: {e}")
        return []

def ensure_experiments_repo() -> pathlib.Path:
    """Clone or update the SWE-bench experiments repository."""
    experiments_path = pathlib.Path(EXPERIMENTS_DIR).expanduser().resolve()
    
    if experiments_path.exists() and (experiments_path / ".git").exists():
        try:
            subprocess.run(["git", "pull"], cwd=experiments_path, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to update repository: {e.stderr}")
            print("Continuing with existing version...")
    else:
        try:
            subprocess.run(["git", "clone", EXPERIMENTS_REPO_URL, str(experiments_path)], check=True, capture_output=True, text=True)
            print("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to clone repository: {e.stderr}")
            sys.exit(1)
    return experiments_path

def count_top_agents_solved_for_split(split_name: str, num_agents: int = 5):
    """
    Generate CSV showing which top agents solved which instances for a specific split.
    
    Args:
        split_name: The split name (e.g., "Verified", "Lite", "Test", "Multimodal")
        num_agents: Number of top agents to analyze
    """
    # Ensure we have the latest experiments repository
    ensure_experiments_repo()
    split_root = Path(EXPERIMENTS_DIR) / "evaluation" / split_name.lower()
    if not split_root.is_dir():
        print(f"ERROR: {split_root} not found. Set EXPERIMENTS_DIR correctly.", file=sys.stderr)
        sys.exit(2)

    lb = read_json_url_or_path(LEADERBOARD_JSON)
    top_agents = top_agents_for_split(lb, split_name, num_agents)

    columns = [] 
    local_union_ids: set[str] = set()

    print(f"Top {num_agents} ({split_name}):")
    for i, row in enumerate(top_agents, 1):
        folder = row.get("folder")
        name = row.get("name") or f"rank{i}"
        print(f"{i:2d}. {name} | resolved={row.get('resolved')} | folder={folder}")
        if not folder:
            continue
        subdir = split_root / folder
        if not subdir.is_dir():
            print(f"[warn] missing folder locally: {subdir}", file=sys.stderr)
            continue
        resolved = read_submission_resolved_ids(subdir)
        columns.append((f"{i:02d}_{folder}", resolved))
        # also collect skipped ids so rows don't get lost
        try:
            j = json.loads((subdir / "results" / "results.json").read_text(encoding="utf-8"))
            local_union_ids |= set(j.get("resolved", []))
            local_union_ids |= set(j.get("no_generation", []))
            local_union_ids |= set(j.get("no_logs", []))
        except Exception:
            pass

    # decide row universe (instance_ids)
    instance_ids = load_canonical_ids_for_split(split_name)
    if not instance_ids:
        instance_ids = sorted(local_union_ids)
        print(f"[warn] using local union of IDs ({len(instance_ids)}). "
              f"For guaranteed canonical IDs, install `datasets` and internet access.", file=sys.stderr)

    out_csv = pathlib.Path(f"top_agents_performance_{split_name.lower()}.csv")

    headers = ["instance_id"] + [col for col, _ in columns] + ["any_top_agents_solved"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        
        non_resolved_instances = []
        
        for iid in instance_ids:
            row_vals = [iid]
            solved_by_any = 0
            
            for _, resolved in columns:
                solved = 1 if iid in resolved else 0
                row_vals.append(solved)
                if solved == 1:
                    solved_by_any = 1
            
            row_vals.append(solved_by_any)
            w.writerow(row_vals)
            
            if solved_by_any == 0:
                non_resolved_instances.append(iid)
    
    print(f"\nResults saved to: {out_csv}")
    print(f"\nNon-resolved instances ({len(non_resolved_instances)}):")
    for instance_id in non_resolved_instances:
        print(f"  {instance_id}")

def find_failed_trajectory(instance_id: str, experiments_dir: str) -> Optional[str]:
    """
    Find a failed trajectory for the given instance within the experiments directory (assume names like '{instance_id}.log').
    If the file exists, return the path to the file.
    Otherwise, call find_failed_trajectory_in_issues_directory to find the failed trajectory.
    """
    trajectory_path = os.path.join(experiments_dir, instance_id + ".log")
    if os.path.exists(trajectory_path):
        return trajectory_path
    else:
        return find_failed_trajectory_in_issues_directory(instance_id, experiments_dir)

def find_successful_trajectory_for_split(instance_id: str, split_name: str, num_agents: int = 5) -> Optional[str]:
    """
    Find a successful trajectory for the given instance in a specific split.
    Checks the top agents performance CSV to find which agents solved the instance, then looks in each agent's directory for trajectory files.
    Tries agents in order of ranking until a trajectory is found.
    Special handling for 20250915_JoyCode agent where trajectories are in instance-named folders.
    Automatically downloads logs and generates CSV if missing.
    
    Args:
        instance_id: The instance ID to find trajectory for
        split_name: The split name (e.g., "Verified", "Lite", "Test", "Multimodal")
        num_agents: Number of top agents to check
    """
    # Check if trajectory folders exist, download if missing
    split_dir = os.path.join(EXPERIMENTS_DIR, "evaluation", split_name.lower())
    if not os.path.exists(split_dir):
        download_top_agents_logs_for_split(EXPERIMENTS_DIR, split_name, num_agents)
    csv_path = os.path.join(os.path.dirname(__file__), f"top_agents_performance_{split_name.lower()}.csv")
    if not os.path.exists(csv_path):
        count_top_agents_solved_for_split(split_name, num_agents)
    
    # Get all agents that solved this instance
    solved_agents = _find_all_agents_for_instance(instance_id, csv_path)
    if not solved_agents:
        print(f"No agent found that solved {instance_id} in {split_name} split")
        return None
    
    # Try each agent until we find a trajectory
    for agent_name in solved_agents:
        print(f"Trying agent {agent_name} for {instance_id}")
        # Special handling for JoyCode agent
        if agent_name == "20250915_JoyCode":
            trajectory_path = _find_joycode_trajectory_for_split(instance_id, EXPERIMENTS_DIR, split_name)
        else:
            trajectory_path = _find_regular_agent_trajectory_for_split(instance_id, agent_name, EXPERIMENTS_DIR, split_name)
        
        if trajectory_path:
            print(f"Found trajectory for {instance_id} using agent {agent_name}")
            return trajectory_path
        else:
            print(f"No trajectory found for {instance_id} using agent {agent_name}")
    
    return None

def _find_best_agent_for_instance(instance_id: str, csv_path: str) -> Optional[str]:
    """
    Find the best agent that solved the given instance by checking the CSV.
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['instance_id'] == instance_id:
                    for col_name in reader.fieldnames:
                        if col_name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_')) and row.get(col_name) == '1':
                            agent_name = col_name[3:]
                            return agent_name
                    return None
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    return None

def _find_all_agents_for_instance(instance_id: str, csv_path: str) -> List[str]:
    """
    Find all agents that solved the given instance by checking the CSV.
    Returns a list of agent names sorted by their ranking (01_, 02_, etc.).
    """
    agents = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['instance_id'] == instance_id:
                    for col_name in reader.fieldnames:
                        if col_name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_')) and row.get(col_name) == '1':
                            agent_name = col_name[3:]
                            agents.append(agent_name)
                    break
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    return agents

def _find_joycode_trajectory_for_split(instance_id: str, experiments_dir: str, split_name: str) -> Optional[str]:
    """
    Find trajectory for JoyCode agent in a specific split (special case with instance-named folders).
    """
    joycode_trajs_dir = os.path.join(experiments_dir, "evaluation", split_name.lower(), "20250915_JoyCode", "trajs")
    instance_dir = os.path.join(joycode_trajs_dir, instance_id)
    if not os.path.exists(instance_dir):
        print(f"JoyCode trajectory directory not found: {instance_dir}")
        return None
    completed_logs_path = os.path.join(instance_dir, "completed_logs.txt")
    if os.path.exists(completed_logs_path):
        return os.path.abspath(completed_logs_path)
    else:
        print(f"completed_logs.txt not found in {instance_dir}")
        return None

def _find_regular_agent_trajectory_for_split(instance_id: str, agent_name: str, experiments_dir: str, split_name: str) -> Optional[str]:
    """
    Find trajectory for regular agents in a specific split (files named with instance_id).
    """
    agent_trajs_dir = os.path.join(experiments_dir, "evaluation", split_name.lower(), agent_name, "trajs")
    if not os.path.exists(agent_trajs_dir):
        print(f"Agent trajectory directory not found: {agent_trajs_dir}")
        return None
    
    try:
        for filename in os.listdir(agent_trajs_dir):
            if instance_id in filename:
                trajectory_path = os.path.join(agent_trajs_dir, filename)
                if os.path.isfile(trajectory_path):
                    return os.path.abspath(trajectory_path)
    except OSError:
        pass    
    return None

def download_single_agent_for_split(agent_info: tuple, experiments_dir: str, split_name: str) -> tuple[str, bool, str]:
    """
    Download logs for a single agent in a specific split. Returns (agent_name, success, error_message).
    """
    i, agent = agent_info
    folder = agent.get("folder")
    name = agent.get("name") or f"rank{i}"
    
    if not folder:
        return (name, False, "No folder specified")
    
    # Check if trajectories already exist
    trajs_path = os.path.join(experiments_dir, "evaluation", split_name.lower(), folder, "trajs")
    if os.path.exists(trajs_path) and os.listdir(trajs_path):
        print(f"[{name}] ✓ Trajectories already exist, skipping download")
        return (name, True, "")
    
    download_path = f"evaluation/{split_name.lower()}/{folder}"
    cmd = [sys.executable, "-m", "analysis.download_logs", "--only_trajs", "--skip_existing", download_path]    
    try:
        result = subprocess.run(cmd, cwd=experiments_dir, check=True)
        print(f"[{name}] ✓ Download completed successfully")
        return (name, True, "")
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"[{name}] ✗ {error_msg}")
        return (name, False, error_msg)

def download_top_agents_logs_for_split(experiments_dir: str = "experiments", split_name: str = "Verified", num_agents: int = 5) -> None:
    """
    Download logs for all top N agents in a specific split using parallel processing.
    
    Args:
        experiments_dir: Directory containing the experiments repository
        split_name: The split name (e.g., "Verified", "Lite", "Test", "Multimodal")
        num_agents: Number of top agents to download logs for (default: 5)
    """
    ensure_experiments_repo()
    print("Fetching leaderboard data...")
    leaderboard_json = read_json_url_or_path(LEADERBOARD_JSON)
    top_agents = top_agents_for_split(leaderboard_json, split_name, num_agents)
    
    print(f"Checking/downloading trajectories for {len(top_agents)} agents in {split_name} split...")
    
    agent_infos = [(i, agent) for i, agent in enumerate(top_agents, 1)]
    
    with ProcessPoolExecutor(max_workers=10) as executor:
        future_to_agent = {
            executor.submit(download_single_agent_for_split, agent_info, experiments_dir, split_name): agent_info[1]
            for agent_info in agent_infos
        }
        
        successful = 0
        failed = 0
        
        for future in as_completed(future_to_agent):
            agent = future_to_agent[future]
            try:
                name, success, error_msg = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                name = agent.get("name", "Unknown")
                print(f"[{name}] ✗ Exception: {e}")
                failed += 1
    print(f"\nDownload completed for {split_name} split. Success: {successful}, Failed: {failed}")