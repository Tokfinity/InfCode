from typing import Any, Dict, List
import json
from traceback import format_exc
from src.managers.log.logger import Logger
from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.prompts.prompts_manager import PromptsManager
from src.managers.loop.base import BaseLoop
from src.tools.base import (
    ToolExecutor,
    ToolResult,
    SubmitToolResult,
    BASH_TOOL_NAME,
    STR_REPLACE_BASED_EDIT_TOOL_NAME,
    SEARCH_TOOL_NAME,
    SUBMIT_RESULT_TOOL_NAME,
)


class PatchGenerator(BaseLoop):
    def __init__(
        self,
        instance_id: str,
        instance_data: Dict[str, Any],
        logger: Logger,
        prompts_manager: PromptsManager | None,
        llm_manager: LLMAPIManager | None,
        tool_executor: ToolExecutor,
        config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(instance_id, instance_data, logger, prompts_manager, llm_manager, tool_executor, config)


    async def _submit_all_tool_calls(
        self, other_tool_calls: List[Dict[str, Any]]
    ) -> List[Any]:
        """execute tool calls, return tool execution results list"""
        if not other_tool_calls:
            return []
        from src.tools.base import ToolCall

        tool_call_objects = []
        for tool_call_dict in other_tool_calls:
            raw_args = tool_call_dict.get("function", {}).get("arguments", {})
            parsed_args = raw_args
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except Exception as e:
                    self.logger.warning(f"[{self.component_name}] In _submit_all_tool_calls function, fail: {e}, traceback: {format_exc()}, args: {raw_args}.")
                    parsed_args = {}
            tool_call_obj = ToolCall(
                name=tool_call_dict.get("function", {}).get("name", ""),
                call_id=tool_call_dict.get("id", ""),
                arguments=parsed_args,
                id=tool_call_dict.get("id", ""),
            )
            tool_call_objects.append(tool_call_obj)
        return await self.tool_executor.container_sequential_tool_call(
            tool_call_objects
        )

    def _process_submit_result_tool_result(
        self,
        submit_result: ToolResult,
        golden_patch: List[Dict[str, Any]],
    ) -> None:
        """process submit_result tool call, fill golden_patch and log"""
        if not submit_result.success or not submit_result.result:
            self.logger.warning(f"[{self.component_name}] submit_result failed and no result.")
            return

        try:
            submit_tool_result = SubmitToolResult.from_string(submit_result.result)

            if submit_tool_result.output:
                patch_info = {
                    "patch_content": submit_tool_result.output,
                    "test_status": submit_tool_result.test_status,
                    "reasoning": submit_tool_result.reasoning,
                }
                golden_patch.clear()
                golden_patch.append(patch_info)
                self.logger.info(
                    f"[{self.component_name}] patch len: {len(submit_tool_result.output)}."
                )
                self.logger.info(
                    f"[{self.component_name}] test status: {submit_tool_result.test_status}."
                )
                self.logger.info(
                    f"[{self.component_name}] reasoning: {submit_tool_result.reasoning[:100]}..."
                )
            else:
                self.logger.warning(
                    f"[{self.component_name}] submit_result success but no patch content."
                )
        except Exception as e:
            self.logger.error(f"[{self.component_name}] parse submit_result result fail: {e}, traceback: {format_exc()}.")

    def _get_tools(self) -> List[Dict[str, Any]]:
        tools = []
        #use_openai_format = self._should_use_openai_format()
        use_openai_format = True

        for tool in self.tool_executor.tools.values():
            if use_openai_format:
                tool_def = tool._definition_for_openai_fmt()
            else:
                tool_def = tool._definition_for_claude_fmt()
            tools.append(tool_def) 

        return tools

    def _should_use_openai_format(self) -> bool:
        
        if not self.llm_manager or not hasattr(self.llm_manager, "get_model_name"):
            return True  # openAI format by default

        model_name = self.llm_manager.get_model_name().lower()
        
        return "claude" not in model_name

    def _get_issue_prompt(self) -> str:
        """generate issue prompt based on instance data"""
        if not self.prompts_manager:
            self.logger.warning("PromptsManager not initialized, cannot generate issue prompt.")
            return ""

        #instance_id = self.instance_data.get("instance_id", "")
        #repo = self.instance_data.get("repo", "")
        created_at = self.instance_data.get("created_at", "")
        base_commit = self.instance_data.get("base_commit", "")
        environment_setup_commit = self.instance_data.get(
            "environment_setup_commit", ""
        )
        version = self.instance_data.get("version", "")
        problem_statement = self.instance_data.get("problem_statement", "")
        difficulty = self.instance_data.get("difficulty", "")

        return self.prompts_manager.format_issue_prompt(
            created_at=created_at,
            base_commit=base_commit,
            environment_setup_commit=environment_setup_commit,
            version=version,
            problem_statement=problem_statement,
            difficulty=difficulty,
        )

    async def _generate_patch(self) -> Dict[str, Any] | None:
        """main loop logic for generating candidate patch"""

        usage_stats = self._init_usage_stats()
        tool_stats = self._init_tools_stats()

        if not self.llm_manager or not self.prompts_manager:
            self.logger.error(f"[{self.component_name}] LLM manager or prompts manager not initialized.")
            return {
                "success": False,
                "golden_patch": [],
                "llm_usage": usage_stats.to_dict(),
                "tool_stats": tool_stats.to_dict(),
                "total_turns": 0,
            }

        tools = self._get_tools()
        
        self._debug_tools(tools)

        root_path = self.config.get("builder", {}).get("repo_root_path", "")
        max_turn = (
            self.config.get("runner", {}).get("generator_loop", {}).get("max_turn", 10)
        )
        temperature = (
            self.config.get("runner", {})
            .get("generator_loop", {})
            .get("temperature", 0.2)
        )

        issue_prompt = self._get_issue_prompt()
        user_prompt = self.prompts_manager.get_generator_user(root_path, issue_prompt)
        system_prompt = self.prompts_manager.get_generator_system(root_path)

        
        total_turns = 0
        golden_patch = []

        try:
            self.logger.info(
                f"[{self.component_name}] {self.instance_id}: start generating candidate patch, max turn: {max_turn}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            self.logger.notice(
                f"[{self.component_name}]: {json.dumps(messages[0], ensure_ascii=False)}"
            )
            self.logger.notice(
                f"[{self.component_name}]: {json.dumps(messages[1], ensure_ascii=False)}"
            )

            for turn in range(max_turn):
                total_turns = turn + 1
                self.logger.info(f"[{self.component_name}] The {total_turns}th turn started.")

                try:
                    current_input_msg = messages[-1] if messages else None
                    if current_input_msg is not None:
                        self.logger.notice(
                            f"[{self.component_name}] The {total_turns}th turn input: {json.dumps(current_input_msg, ensure_ascii=False)}"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"[{self.component_name}] {total_turns}th turn: LLM input fail: {messages[-1] if messages else None}, error: {e}, traceback: {format_exc()}."
                    )

                self._debug_messages(turn, messages)

                response = self.llm_manager.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                )

                first_content: str = ""
                first_tool_calls: Any = None
                if hasattr(response, "choices") and response.choices:
                    ch0 = response.choices[0]
                    first_content = (
                        getattr(getattr(ch0, "message", None), "content", None) or ""
                    )
                    first_tool_calls = getattr(
                        getattr(ch0, "message", None), "tool_calls", None
                    )

                self._response_log(
                    response, first_content, first_tool_calls, total_turns
                )

                self._update_usage(response, usage_stats)

                if hasattr(response, "choices") and response.choices:
                    content = first_content
                    tool_calls = first_tool_calls

                    if not self._make_assistant(content, tool_calls, messages):
                        continue

                    if tool_calls:
                        self.logger.info(
                            f"[{self.component_name}] {total_turns}th turn: call {len(tool_calls)} tools."
                        )

                        tool_results = await self._submit_all_tool_calls(tool_calls)

                        self._update_tool_call_statistic(tool_results, tool_stats)

                        if tool_results:
                            submit_result = None
                            other_tool_results = []

                            for tool_result in tool_results:
                                tool_name = getattr(tool_result, "name", "")
                                if tool_name == SUBMIT_RESULT_TOOL_NAME:
                                    submit_result = tool_result
                                else:
                                    other_tool_results.append(tool_result)

                            if submit_result:
                                self.logger.debug(
                                    f"[{self.component_name}] {total_turns}th turn: got submit_result tool call."
                                )
                                self.logger.debug(f"[{self.component_name}] {total_turns}th turn: submit_result result: {submit_result}")
                        
                                self._process_submit_result_tool_result(
                                    submit_result, golden_patch
                                )
                                
                                self._debug_last_message(turn, messages)
                                break

                            if other_tool_results:
                                self._make_tool_response(other_tool_results, messages)

                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": "请继续分析问题并使用工具来解决问题。",
                            }
                        )

            self.logger.debug(f"[{self.component_name}] final golden_patch: {golden_patch}")
        
            success = (
                len(golden_patch) > 0 and golden_patch[0].get("patch_content", "") != ""
            )

            self.logger.info(
                f"[{self.component_name}] status={success}, total_turns={total_turns}, tools_stats={tool_stats}"
            )

            result_payload = {
                "success": success,
                "golden_patch": golden_patch,
                "llm_usage": usage_stats.to_dict(),
                "tool_stats": tool_stats.to_dict(),
                "total_turns": total_turns,
            }
            try:
                self.logger.notice(
                    f"[{self.component_name}] final output: {json.dumps(result_payload, ensure_ascii=False)}"
                )
            except Exception as e:
                self.logger.warning(
                    f"[{self.component_name}] output: {str(result_payload)}, error: {e}, traceback: {format_exc()}."
                )
            return result_payload

        except Exception as e:
            self.logger.error(f"[{self.component_name}] fail: {e}, traceback: {format_exc()}.")
            result_payload = {
                "success": False,
                "golden_patch": [],
                "llm_usage": usage_stats.to_dict(),
                "tool_stats": tool_stats.to_dict(),
                "total_turns": total_turns,
            }
            try:
                self.logger.notice(
                    f"[{self.component_name}] 最终返回数据(失败): {json.dumps(result_payload, ensure_ascii=False)}"
                )
            except Exception as e:
                self.logger.notice(
                    f"[{self.component_name}] 最终返回数据(失败, 字符串回退): {str(result_payload)}, error: {e}, traceback: {format_exc()}."
                )
            return result_payload
