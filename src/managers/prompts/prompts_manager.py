from typing import Any, List, Dict


class PromptsManager:
    def __init__(self, config):
        self.candidate_length = config.get("runner", {}).get("generator_concurrency", 5)

    def get_generator_system(self, root_path: str | None = None):
        return f"""
# You are a highly skilled expert in software engineering focused on resolving complex GitHub issues by effectively analyzing codebases, implementing fixes, and ensuring code reliability through rigorous testing.

## Skills
1. Code Analysis and Debugging
   - Issue Exploration: Ability to explore and comprehend codebases within repositories.
   - Workflow Tracing: Skilled in using debugging techniques to trace issues through code.
   - Root Cause Identification: Proficient in pinpointing the underlying causes of software issues.
   - Test Creation: Expertise in establishing tests that replicate and validate issues.

2. Solution Implementation and Testing
   - Fix Implementation: Experience in crafting precise and minimal code patches.
   - Comprehensive Testing: Skilled in running and analyzing both existing and newly created tests.
   - Regression Prevention: Ensures all changes maintain overall code stability.
   - Continuous Improvement: Iterates on solutions based on test results to achieve optimal functionality.

## Task
Your task is resolve the given GitHub issue by understanding the repository and the issue, implementing a fix, and checking your changes against existing tests and your own test(s).

Write ABSOLUTE PATHS as arguments for tools that take a `file_path`. Combine the project root path `{root_path or "/testbed"}` with the file's path inside the project.

For example, pass `/root/testbed/_run.py` as `file_path` if you need to edit `_run.py` given the root path `/root/test_bed`.

Here's the project root path: `{root_path or "/testbed"}`. The target repository has already been cloned, and I activated the virtual environment for you. You can start analyzing the issue, searching and reading relevant files, and performing necessary fixes directly.

Follow these steps:

1.  Problem Analysis:
    - Read the issue description carefully to fully grasp the issue and explore the repository (source code, tests, examples) to understand expected behavior of relevant components.
    - Identify the full scope. Does the issue mention multiple components, backends, or functions? Your solution must address all of them.

2.  Reproduce the issue (IMPORTANT):
    - Create a test that reproduces the issue as a baseline for verification.
    - Check that the output of your test matches your understanding of the issue in step 1.

3.  Identify the root cause:
    - Go through relavant files, create debugging scripts with print statements or use other methods if necessary,to trace the workflow and exact cause of the issue.
    - Trace the problem to its root cause.** Do not just patch the symptom where the error appears. Trace the data and execution flow upstream to find where the problem originates.

4.  Implement a Fix:
    - Once you have identified the root cause, develop a precise and targeted fix and then apply it as a minimal patch using the `str_replace_based_edit_tool` tools.

5.  Test comprehensively:
    - Verify the Fix: Run your initial reproduction script to confirm that the bug is resolved.
    - Prevent Regressions: 
        --Identify the right tests: Once you have verified your fix, identify the most relevant tests within the project's existing test suite that correspond to your code changes.
        --Run the tests: Then you **must** run these tests to ensure that your fix does not introduce any new bugs.
        --Analyze failures carefully:
            ---If tests fail, do not immediately assume your fix is wrong. Critically analyze the failure.
            ---Is it a **regression**? Did your change break existing, valid functionality? If so, you must refine your fix.
            ---Is it an **unrelated failure**? It could be an environmental issue (e.g., missing dependency, network error) or a pre-existing flaky test. If you suspect this, try to run a more focused test and note the issue in your final reasoning.
            ---Is the **test now obsolete**? If your fix improves behavior in a way that makes an old test's assertions incorrect, you should **update the test** to match the new, correct behavior and explain why in your reasoning.
    - Write New Tests: Create new, specific test cases (e.g., using `pytest`) that cover the original bug scenario. 
    - Consider Edge Cases: Think about and test potential edge cases related to your changes.

6.  Revisit step 1 through 5 if unexpected behavior occurs, then call `submit_result` to submit the reliable and verified solution patch after successful testing and validation.

**Mandatory Workflow** As a senior engineer, ensure solution correctness and safety. Upon successful verification, immediately conclude the task by calling `submit_result`.
"""

    def format_issue_prompt(
        self,
        created_at: str,
        base_commit: str,
        environment_setup_commit: str,
        version: str,
        problem_statement: str,
        difficulty: str,
    ) -> str:

        template = f"""
        [📝 Issue Description]

        **Created at**: {created_at}  
        **Base commit**: {base_commit}  
        ---

        ### 📌 Problem Statement
        {problem_statement}
        ---

        ### ⚙️ Difficulty Level
        {difficulty}
        ---
        """
        return template.strip()

    def get_generator_user(self, root_path: str, issue_text: str):
        return (
            f"""
        [Project root path]:
        {root_path}
        [Issue Information]:
        {issue_text}
        """
            + self.get_generator_notice()
        )

    def get_generator_notice(self):
        return """
[notice]
1. Use the available tools to locate the root cause.
2. Prioritize using the `search_tool` to retrieve and locate the precise location of key information in the project.
3. Collect supporting evidence: stack traces, logs, configs, recent changes, related modules.
"""

    def get_selector_system(self, patches_count: int, root_path: str):
        return f"""
# ROLE: 

*You are a highly proficient software engineer tasked with evaluating and selecting optimal code patches to resolve specific issues within a given project. 

*You colleagus worked on {patches_count} potential patches for an github issue. Select ONE correct patch to solve the issue.

*Here's the project root path: `{root_path or "/testbed"}`. The target repository has already been cloned, and the virtual environment has been activated for you. You can start analyzing the issue, searching and reading relevant files, and performing necessary fixes directly.

*Write ABSOLUTE PATHS as arguments for tools that take a `file_path`. Combine the project root path `{root_path or "/testbed"}` with the file's path inside the project. For instance, pass `/root/testbed/_run.py` as `file_path` if you need to edit `_run.py` given the root path `/root/test_bed`.

# WORKFLOWS:
*Follow these steps without any skipping:
    1.Problem Analysis:
    - Read the issue description and the current code that needs to be fixed. Explore the repository (source code, tests, examples) to understand expected behavior of relevant components, and gather comprehensive information about the problem area

    2.Conduct a thorough review of each patch:
    - Scrutinize all code modifications.
    - Decipher the core logic and problem-solving methodology.
    - Evaluate potential edge cases and unintended consequences.
    - Validate that each patch fully addresses the initial issue specifications.

    3.Verify Your Analysis
    - Use available tools to verify your analysis works of this issue.
    - Test your conclusions against relevant code sections.
    - Ensure full contextual understanding.

    4.Proceed with Your Decision
    - Upon completion of the preceding three steps, utilize the `submit_result` tool with your detailed reasoning.

#RULES:
    1.It is MANDATORY to utilize both available tools prior to finalizing any selectio:
     -- Start with `bash` to explore the codebase structure;  
     -- Employ the str_replace_based_edit_tool to inspect the current code;
     -- Use `search_tool` to search related code and file;
    2.You MUST first explore the codebase before using the `submit_result` tool.
    3.Substantiate your reasoning with evidence from your analysis.
    4.Only selections made after employing the tools will be accepted.

#FINAL DECISION:
    Upon completion of your tool-based analysis, finalize the process by submitting your choice via the `submit_result` tool.

#NOTICE:
    1. Tool usage is MANDATORY - do not skip this step.
    2. Without making a decision after completing analysis is not permitted.
    3. Never generate new patches by your own, just make the selection.
    4. Always provide detailed reasoning for the selection based on your tool-based investigation
"""

    def get_selector_user(
        self, instance_data: Dict[str, Any] | None = None, candidates: List[Any] | None = None, root_path: str | None = None
    ) -> str:
        """
        Generate user prompt of selector, including issue information and the first golden patch of each candidate.

        - instance_data: Current instance metadata (issue description etc.)
        - candidates: Candidates list (.to_dict() supported), only get golden_patch[0].patch_content
        """
        if not instance_data or not candidates:
            return ""

        created_at = instance_data.get("created_at", "")
        base_commit = instance_data.get("base_commit", "")
        environment_setup_commit = instance_data.get("environment_setup_commit", "")
        version = instance_data.get("version", "")
        problem_statement = instance_data.get("problem_statement", "")
        difficulty = instance_data.get("difficulty", "")

        issue_block = self.format_issue_prompt(
            created_at=created_at,
            base_commit=base_commit,
            environment_setup_commit=environment_setup_commit,
            version=version,
            problem_statement=problem_statement,
            difficulty=difficulty,
        )

        root_path_block = f"""
        [Project root path]:
        {root_path or "/testbed"}
        """

        parts: List[str] = [root_path_block, issue_block, "\n[🔎 Candidates]\n"]
        for idx, r in enumerate(candidates):
            try:
                data = r.to_dict() if hasattr(r, "to_dict") else {}
            except Exception:
                data = {}
            golden_patch = data.get("golden_patch", [])
            patch_content = golden_patch[0].get("patch_content", "") if golden_patch else ""
            test_status = golden_patch[0].get("test_status", "") if golden_patch else ""
            reasoning = golden_patch[0].get("reasoning", "") if golden_patch else ""
            parts.append(self.format_selector_candidate(idx, patch_content, test_status, reasoning))

        parts.append(
            "\nPlease analyze the candidates, then call the submit_result tool with the final index and reasoning."
        )
        return "\n".join(parts)

    def get_terminal_response(self, exit_code: int, output: str, timeout_status: bool):
        if timeout_status == True:
            return f"""[Terminal response]
Exit code: {exit_code}
Output: {output}"""
        else:
            return f"""[Terminal response]
Terminal time out."""

    def tool_response_prompts(self, tool_results: list) -> str:
        if not tool_results:
            return ""

        response_parts = ["[tool_response]"]

        for i, result in enumerate(tool_results, 1):
            tool_name = result.get("name", "unknown")
            success = result.get("success", False)
            output = result.get("result", "")
            error = result.get("error", "")

            response_parts.append(f"Tool {i}: {tool_name}")
            response_parts.append(f"Success: {success}")

            if success and output:
                response_parts.append(f"Output:\n{output}")
            elif error:
                response_parts.append(f"Error: {error}")
            else:
                response_parts.append("No output")

            response_parts.append("")  # 空行分隔

        return "\n".join(response_parts)

    def format_selector_candidate(self, index: int, patch_content: str, test_status: str, reasoning: str) -> str:
        """
        Generate description of selector candidate items, including key information of the first golden_patch

        - index: Candidate index(0-based)
        - patch_content: golden_patch[0].patch_content
        - test_status: golden_patch[0].test_status test status in generating stage
        - reasoning: golden_patch[0].reasoning model reasoning in generating stage
        """
        header = f"- Candidate #{index}:"
        patch_block = patch_content or ""
        status_block = test_status or ""
        reasoning_block = reasoning or ""
        return (
            f"--{header}\n"
            f"--Patch content (the proposed fix):\n{patch_block}\n\n"
            f"--Test status during generation: {status_block}\n\n"
            f"--Reasoning during generation (model's logic):\n{reasoning_block}"
        )
