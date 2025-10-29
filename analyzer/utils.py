import urllib.request
import json
import pathlib
import os
import re
from typing import Optional


def read_json_url_or_path(src: str) -> dict:
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=60) as r:
            return json.load(r)
    p = pathlib.Path(src).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

# an "issues" directory containing "{numerical_id}-{instance_id}/logs/{timestamp}/run/{instance_id}/generator/{generator_id}/" and within it "notice.log" and "debug.log"
def find_failed_trajectory_in_issues_directory(instance_id: str, experiments_dir: str) -> Optional[str]:
    """
    Find a failed trajectory for the given instance within the experiments directory.
    
    The function searches for debug.log files that contain successful patches and returns
    the path to the corresponding notice.log file.
        
    The function searches through the directory structure:
    {experiments_dir}/{numerical_id}-{instance_id}/logs/{timestamp}/run/{instance_id}/generator/{generator_id}/
    """
    if not os.path.exists(experiments_dir):
        return None
    pattern = re.compile(r'^\d+-' + re.escape(instance_id) + r'$')
    for item in os.listdir(experiments_dir):
        if pattern.match(item):
            instance_dir = os.path.join(experiments_dir, item)
            if not os.path.isdir(instance_dir):
                continue
            logs_dir = os.path.join(instance_dir, 'logs')
            if not os.path.exists(logs_dir):
                continue
            timestamp_dirs = []
            for timestamp in os.listdir(logs_dir):
                timestamp_path = os.path.join(logs_dir, timestamp)
                if os.path.isdir(timestamp_path):
                    timestamp_dirs.append(timestamp)
            timestamp_dirs.sort(reverse=True)
            for timestamp in timestamp_dirs:
                timestamp_path = os.path.join(logs_dir, timestamp)
                run_dir = os.path.join(timestamp_path, 'run', instance_id, 'generator')
                if not os.path.exists(run_dir):
                    continue
                for generator_id in range(5):
                    generator_dir = os.path.join(run_dir, f"{generator_id:03d}")
                    debug_log_path = os.path.join(generator_dir, 'debug.log')
                    notice_log_path = os.path.join(generator_dir, 'notice.log')
                    if not os.path.exists(debug_log_path):
                        continue                    
                    if _contains_successful_patch(debug_log_path):
                        if os.path.exists(notice_log_path):
                            return os.path.abspath(notice_log_path)
    return None

def _contains_successful_patch(debug_log_path: str) -> bool:
    """
    Check if a debug.log file contains a successful patch.
    
    Reads the last non-empty line, splits on 'result_data: ', parses the dict,
    and checks if the golden_patch contains non-empty patch_content.
    """
    try:
        with open(debug_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()        
        last_line = None
        for line in reversed(lines):
            if line.strip():
                last_line = line.strip()
                break
        if not last_line:
            return False
        if 'result_data: ' not in last_line:
            return False        
        parts = last_line.split('result_data: ', 1)
        if len(parts) != 2:
            return False
        dict_str = parts[1]        
        try:
            result_data = eval(dict_str)
        except:
            return False        
        if not isinstance(result_data, dict):
            return False        
        golden_patch = result_data.get('golden_patch')
        if not isinstance(golden_patch, list) or len(golden_patch) == 0:
            return False        
        first_patch = golden_patch[0]
        if not isinstance(first_patch, dict):
            return False        
        patch_content = first_patch.get('patch_content')
        if not isinstance(patch_content, str) or not patch_content.strip():
            return False
        return True
    except (IOError, UnicodeDecodeError, Exception):
        return False