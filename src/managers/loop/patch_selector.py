from typing import Any, Dict, List
import json
from traceback import format_exc
from src.managers.log.logger import Logger
from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.prompts.prompts_manager import PromptsManager
from src.managers.loop.types import GeneratorResult, SelectorResult, LLMUsage, ToolStats, PatchInfo
from src.tools.base import ToolExecutor, ToolCall, ToolResult
from src.managers.loop.base import BaseLoop

SELECTOR_SUBMIT_TOOL_NAME = "submit_result"

class PatchSelector(BaseLoop):
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

    def _get_submit_result_tool_name(self):
        return SELECTOR_SUBMIT_TOOL_NAME

    def _definition_for_submit_tool(self, use_openai_format: bool) -> Dict[str, Any]:
        """submit_result tool"""
        if use_openai_format:
            return {
                "type": "function",
                "function": {
                    "name": self._get_submit_result_tool_name(),
                    "description": "Submit the final selected patch index and reasoning.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "The chosen patch index (0-based).",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Detailed reasoning for the selection.",
                            },
                        },
                        "required": ["index", "reason"],
                    },
                },
            }
        
        return {
            "type": "function",
            "function": {
                "name": self._get_submit_result_tool_name(),
                "description": "Submit the final selected patch index and reasoning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "The chosen patch index (0-based).",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Detailed reasoning for the selection.",
                        },
                    },
                    "required": ["index", "reason"],
                },
            },
        }

    def _build_user_prompt(self, candidates: List[GeneratorResult], root_path: str) -> str:
        
        if not self.prompts_manager:
            return ""
        return self.prompts_manager.get_selector_user(self.instance_data, candidates, root_path)

    def _get_system_prompt(self, patches_count: int, root_path: str) -> str:
        if not self.prompts_manager:
            return ""
        return self.prompts_manager.get_selector_system(patches_count, root_path)

    def _get_tools(self) -> List[Dict[str, Any]]:
        
        tool_defs: List[Dict[str, Any]] = []
        try:
            for tool in self.tool_executor.tools.values():
                try:
                    tool_defs.append(tool._definition_for_openai_fmt())
                except Exception:
                    
                    continue
        except Exception:
            pass
        
        tool_defs.append(self._definition_for_submit_tool(True))
        return tool_defs

    def _extract_submit_choice(self, tool_call: Dict[str, Any]) -> Dict[str, Any] | None:
        
        if not tool_call:
            return None
        fn = tool_call.get("function", {})
        if fn.get("name") != self._get_submit_result_tool_name():
            return None
        raw_args = fn.get("arguments", {})
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {}
        index = args.get("index")
        reason = args.get("reason")
        if isinstance(index, int) and index >= 0:
            return {"index": index, "reason": reason or ""}
        return None

    async def _submit_other_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        
        if not tool_calls:
            return []
        to_run: List[ToolCall] = []
        for tool_call_dict in tool_calls:
            fn = tool_call_dict.get("function", {})
            name = fn.get("name", "")
            if name == SELECTOR_SUBMIT_TOOL_NAME:
                continue
            raw_args = fn.get("arguments", {})
            parsed_args = raw_args
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except Exception:
                    parsed_args = {}
            to_run.append(
                ToolCall(
                    name=name,
                    call_id=tool_call_dict.get("id", ""),
                    arguments=parsed_args,
                    id=tool_call_dict.get("id", ""),
                )
            )
        if not to_run:
            return []
        results: List[ToolResult] = await self.tool_executor.container_sequential_tool_call(to_run)
        return results

    async def _select_patch(self, candidates: List[GeneratorResult]) -> SelectorResult:
       
        if not candidates:
            raise ValueError("No candidates provided")
        if not self.llm_manager:
            raise ValueError("LLM manager is not initialized")

        
        tools = self._get_tools()
        
        self._debug_tools(tools)

        root_path = self.config.get("builder", {}).get("repo_root_path", "")
        system_prompt = self._get_system_prompt(len(candidates), root_path)
        user_prompt = self._build_user_prompt(candidates, root_path)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        
        try:
            self.logger.notice(
                f"[{self.component_name}]: {json.dumps(messages[0], ensure_ascii=False)}"
            )
            self.logger.notice(
                f"[{self.component_name}]: {json.dumps(messages[1], ensure_ascii=False)}"
            )
        except Exception:
            self.logger.warning(
                f"[{self.component_name}] Initial fail in selector loop: SP={str(messages[0])}, UP={str(messages[1])}, traceback: {format_exc()}."
            )

        max_turn = int(
            self.config.get("runner", {})
            .get("selector_loop", {})
            .get("max_turn", 200)
        )
        temperature = (
            self.config.get("runner", {})
            .get("selector_loop", {})
            .get("temperature", 0.2)
        )

        usage_stats = self._init_usage_stats()
        tool_stats = self._init_tools_stats()
        
        total_turns = 0

        chosen_index: int | None = None
        select_reason: str = ""
        for turn in range(max_turn):
            try:
                try:
                    current_input_msg = messages[-1] if messages else None
                    if current_input_msg is not None:
                        self.logger.notice(
                            f"[{self.component_name}] The {turn+1}th turn input: {json.dumps(current_input_msg, ensure_ascii=False)}"
                        )
                except Exception:
                    self.logger.warning(
                        f"[{self.component_name}] {turn+1}th turn fail: {messages[-1] if messages else None}, traceback: {format_exc()}."
                    )

                self._debug_messages(turn, messages)

                response = self.llm_manager.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                )

                first_tool_calls = None
                if hasattr(response, "choices") and response.choices:
                    ch0 = response.choices[0]
                    first_tool_calls = getattr(getattr(ch0, "message", None), "tool_calls", None)
                    first_content = getattr(getattr(ch0, "message", None), "content", None) or ""
                else:
                    first_content = ""

                total_turns = turn + 1
                
                self._response_log(response, first_content, first_tool_calls, turn + 1)
                
                self._update_usage(response, usage_stats)

                
                if first_tool_calls:
                    
                    if not self._make_assistant(first_content, first_tool_calls, messages):
                        messages.append(
                            {
                                "role": "user",
                                "content": "请完成分析并调用 submit_result 工具给出最终选择与理由。",
                            }
                        )
                        continue
                    submit_found = False
                    for tc in first_tool_calls:
                        choice = self._extract_submit_choice(tc)
                        if choice is not None:
                            chosen_index = choice["index"]
                            reason = choice.get("reason", "")
                            self.logger.info(
                                f"[{self.component_name}] choose: index={chosen_index}, reason={reason}"
                            )
                            select_reason = reason or ""
                            submit_found = True
                            
                            self._debug_last_message(turn, messages)
                            break
                    
                    
                    if not submit_found:
                        
                        results = await self._submit_other_tool_calls(first_tool_calls)
                        
                        self._make_tool_response(results, messages)
                        
                        self._update_tool_call_statistic(results, tool_stats)

                else:
                    
                    messages.append(
                        {
                            "role": "user",
                            "content": "请完成分析并调用 submit_result 工具给出最终选择与理由。",
                        }
                    )
                
                if chosen_index is not None:
                    break

            except Exception as e:
                self.logger.warning(
                    f"[{self.component_name}] fail: {e}, traceback: {format_exc()}"
                )
                break

        if chosen_index is None:
            # If the model provides no choice, fallback: pick the first successful one; otherwise the first
            for i, r in enumerate(candidates):
                try:
                    if r.success:
                        chosen_index = i
                        break
                except Exception:
                    continue
            if chosen_index is None:
                chosen_index = 0

        if not (0 <= chosen_index < len(candidates)):
            chosen_index = 0

        selected = candidates[chosen_index]
        
        try:
            gp = selected.golden_patch[0] if selected.golden_patch else None
            if gp is None:
                patch_info = PatchInfo(patch_content="", test_status="", reasoning="")
            else:
                patch_info = PatchInfo(
                    patch_content=gp.patch_content,
                    test_status=gp.test_status,
                    reasoning=gp.reasoning,
                )
        except Exception:
            patch_info = PatchInfo(patch_content="", test_status="", reasoning="")

        selector_result = SelectorResult(
            instance_id=selected.instance_id,
            generator_id=selected.generator_id,
            image=selected.image,
            success=True,
            golden_patch=patch_info,
            llm_usage=usage_stats,
            tool_stats=tool_stats,
            total_turns=total_turns,
            select_reason=select_reason,
            error=None,
        )
        return selector_result


