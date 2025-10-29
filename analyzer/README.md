# SWE-Bench Trajectory Analysis

A comprehensive analysis tool for comparing failed and successful software engineering trajectories across different SWE-bench dataset splits. This tool helps identify patterns in agent failures and provides actionable insights for improving software engineering agents.

## Features

- **Multi-Split Support**: Analyze trajectories across SWE-bench splits (Verified, Lite, Test, Multimodal)
- **Automated Comparison**: Compare your agent's failed trajectories with successful trajectories from top-performing agents
- **LLM-Powered Analysis**: Uses advanced language models to generate detailed failure analysis reports
- **Batch Processing**: Process multiple instances in parallel for balanced analysis
- **Leaderboard Integration**: Automatically downloads and analyzes top agent results from SWE-bench leaderboards

## Requirements

- Python 3.8+
- Required packages (see `requirements.txt`):
  - `datasets>=2.0.0` - For loading SWE-bench datasets
  - `requests>=2.25.0` - For API calls
  - `python-dotenv>=0.19.0` - For environment variable management
  - `PyYAML>=6.0` - For configuration file parsing
  - `boto3>=1.40.52` - For AWS S3 access

## Installation

1. Clone the repository and navigate to the directory:
   ```bash
   cd trajectory_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

## Configuration

Edit `config.yaml` to configure your analysis:

```yaml
# Dataset configuration
dataset:
  split: "Verified"  # Options: Verified, Lite, Test, Multimodal

# Paths
paths:
  txt_path: "/path/to/unsolved_instances.txt"
  intermediate_output_path: "/path/to/intermediate_output/"
  output_path: "/path/to/final_report.txt"
  logs_directory: "/path/to/failed_trajectories/"
  prompts_path: "/path/to/prompts_manager.py"
  tools_path: "/path/to/tools/"

# Settings
settings:
  max_workers: 20 # Maximum number of parallel requests to OpenRouter
  num_agents: 4 # Number of top SWE-bench leaderboard agents' to download trajectories for
```

## Usage

### Basic Usage

1. **Prepare your data**:
   - Create a text file listing unresolved instance IDs (one per line)
   - Ensure your failed trajectories are in the specified logs directory

2. **Run the analysis**:
   ```bash
   python run.py
   ```

### Input Requirements

- **Failed Trajectories**: Your agent's failed trajectories in one of these formats:
  - Direct `.log` files named `{instance_id}.log`
  - Issues directory structure: `issues/{numerical_id}-{instance_id}/logs/{timestamp}/run/{instance_id}/generator/{generator_id}/`

- **Unresolved Instances**: Text file containing instance IDs that your agent failed to solve

### Output

The tool generates:
- **Individual Analysis**: Detailed failure analysis for each instance in the intermediate output directory
- **Comprehensive Report**: Categorized analysis of all failures with improvement recommendations

## Architecture

### Core Components

- **`run.py`**: Main entry point and batch processing coordinator
- **`trajectory_discovery.py`**: Handles trajectory finding and leaderboard integration
- **`pairwise_analysis.py`**: LLM-powered failure analysis and dataset loading
- **`utils.py`**: Utility functions for file operations and trajectory parsing
- **`config.yaml`**: Configuration management

## Supported SWE-bench Splits

| Split | Dataset | Instances | Description |
|-------|---------|-----------|-------------|
| Verified | `princeton-nlp/SWE-bench_Verified` | 500 | Verified high-quality instances |
| Lite | `SWE-bench/SWE-bench_Lite` | 300 | Smaller subset for faster testing |
| Test | `SWE-bench/SWE-bench` | 2,294 | Full test set |
| Multimodal | `SWE-bench/SWE-bench_Multimodal` | 510 | Instances with UI elements |

## Error Handling

The tool includes robust error handling for:
- Missing trajectory files
- API rate limiting and failures
- Dataset loading errors
- Invalid configuration parameters

# Contributor

- [Hengzhi Zhang](https://henryzhang11.github.io)

## License

This project is licensed under the MIT License - see the project LICENSE file for details.