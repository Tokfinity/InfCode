"""
This module defines the GeneratorResult data structure for patch generation results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from src.tools.base import (
    BASH_TOOL_NAME,
    STR_REPLACE_BASED_EDIT_TOOL_NAME,
    SEARCH_TOOL_NAME,
    SUBMIT_RESULT_TOOL_NAME,
)


@dataclass
class LLMUsage:
    """LLM usage statistics."""

    prompt_tokens: int = 0  
    completion_tokens: int = 0  
    total_tokens: int = 0  

    def to_dict(self) -> Dict[str, int]:
        """Serialize LLMUsage to a plain dictionary."""
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
        }


@dataclass
class ToolStats:
    """Tool usage statistics per tool.

    Each tool is represented by a small map with two fields:
    - count: total invocation count
    - failed: failed invocation count
    """

    bash: Dict[str, int] = field(default_factory=lambda: {"count": 0, "failed": 0})
    edit: Dict[str, int] = field(default_factory=lambda: {"count": 0, "failed": 0})
    search: Dict[str, int] = field(default_factory=lambda: {"count": 0, "failed": 0})
    submit_result: Dict[str, int] = field(default_factory=lambda: {"count": 0, "failed": 0})

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Serialize ToolStats to a plain dictionary."""
        return {
            BASH_TOOL_NAME: {"count": int(self.bash.get("count", 0)), "failed": int(self.bash.get("failed", 0))},
            STR_REPLACE_BASED_EDIT_TOOL_NAME: {"count": int(self.edit.get("count", 0)), "failed": int(self.edit.get("failed", 0))},
            SEARCH_TOOL_NAME: {"count": int(self.search.get("count", 0)), "failed": int(self.search.get("failed", 0))},
            SUBMIT_RESULT_TOOL_NAME: {"count": int(self.submit_result.get("count", 0)), "failed": int(self.submit_result.get("failed", 0))},
        }


@dataclass
class PatchInfo:
    """Information about a generated patch."""

    patch_content: str  
    test_status: str 
    reasoning: str 


@dataclass
class GeneratorResult:
    """Result from a patch generator."""

    instance_id: str
    generator_id: int
    image: str
    success: bool
    golden_patch: List[
        PatchInfo
    ] 
    llm_usage: LLMUsage
    tool_stats: ToolStats
    total_turns: int
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratorResult":
        """Create GeneratorResult from dictionary."""
        # Handle golden_patch conversion
        golden_patch = []
        if data.get("golden_patch"):
            for patch_data in data["golden_patch"]:
                if isinstance(patch_data, dict):
                    golden_patch.append(
                        PatchInfo(
                            patch_content=patch_data.get("patch_content", ""),
                            test_status=patch_data.get("test_status", ""),
                            reasoning=patch_data.get("reasoning", ""),
                        )
                    )
                else:
                    # Legacy format: just patch content string
                    golden_patch.append(
                        PatchInfo(
                            patch_content=str(patch_data), test_status="", reasoning=""
                        )
                    )

        # Handle LLM usage
        llm_usage_data = data.get("llm_usage", {})
        llm_usage = LLMUsage(
            prompt_tokens=llm_usage_data.get("prompt_tokens", 0),
            completion_tokens=llm_usage_data.get("completion_tokens", 0),
            total_tokens=llm_usage_data.get("total_tokens", 0),
        )

        # Handle tool stats
        tool_stats_data = data.get("tool_stats", {})
        tool_stats = ToolStats(
            bash=tool_stats_data.get(BASH_TOOL_NAME, 0),
            edit=tool_stats_data.get(STR_REPLACE_BASED_EDIT_TOOL_NAME, 0),
            search=tool_stats_data.get(SEARCH_TOOL_NAME, 0),
            submit_result=tool_stats_data.get(SUBMIT_RESULT_TOOL_NAME, 0),
        )

        return cls(
            instance_id=data.get("instance_id", ""),
            generator_id=data.get("generator_id", 0),
            image=data.get("image", ""),
            success=data.get("success", False),
            golden_patch=golden_patch,
            llm_usage=llm_usage,
            tool_stats=tool_stats,
            total_turns=data.get("total_turns", 0),
            error=data.get("error"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert GeneratorResult to dictionary."""
        return {
            "instance_id": self.instance_id,
            "generator_id": self.generator_id,
            "image": self.image,
            "success": self.success,
            "golden_patch": [
                {
                    "patch_content": patch.patch_content,
                    "test_status": patch.test_status,
                    "reasoning": patch.reasoning,
                }
                for patch in self.golden_patch
            ],
            "llm_usage": {
                "prompt_tokens": self.llm_usage.prompt_tokens,
                "completion_tokens": self.llm_usage.completion_tokens,
                "total_tokens": self.llm_usage.total_tokens,
            },
            "tool_stats": {
                BASH_TOOL_NAME: self.tool_stats.bash,
                STR_REPLACE_BASED_EDIT_TOOL_NAME: self.tool_stats.edit,
                SEARCH_TOOL_NAME: self.tool_stats.search,
                SUBMIT_RESULT_TOOL_NAME: self.tool_stats.submit_result,
            },
            "total_turns": self.total_turns,
            "error": self.error,
        }


@dataclass
class SelectorResult:
    """Result from a patch selector.
    """

    instance_id: str
    generator_id: int
    image: str
    success: bool
    golden_patch: PatchInfo
    llm_usage: LLMUsage
    tool_stats: ToolStats
    total_turns: int
    select_reason: str
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectorResult":
        """Create SelectorResult from dictionary."""

        gp_data = data.get("golden_patch", {})
        if isinstance(gp_data, dict):
            golden_patch = PatchInfo(
                patch_content=gp_data.get("patch_content", ""),
                test_status=gp_data.get("test_status", ""),
                reasoning=gp_data.get("reasoning", ""),
            )
        else:
            golden_patch = PatchInfo(
                patch_content=str(gp_data) if gp_data is not None else "",
                test_status="",
                reasoning="",
            )

        # LLM usage
        llm_usage_data = data.get("llm_usage", {})
        llm_usage = LLMUsage(
            prompt_tokens=llm_usage_data.get("prompt_tokens", 0),
            completion_tokens=llm_usage_data.get("completion_tokens", 0),
            total_tokens=llm_usage_data.get("total_tokens", 0),
        )

        # Tool stats
        tool_stats_data = data.get("tool_stats", {})
        tool_stats = ToolStats(
            bash=tool_stats_data.get(BASH_TOOL_NAME, 0),
            edit=tool_stats_data.get(STR_REPLACE_BASED_EDIT_TOOL_NAME, 0),
            search=tool_stats_data.get(SEARCH_TOOL_NAME, 0),
            submit_result=tool_stats_data.get(SUBMIT_RESULT_TOOL_NAME, 0),
        )

        return cls(
            instance_id=data.get("instance_id", ""),
            generator_id=data.get("generator_id", 0),
            image=data.get("image", ""),
            success=data.get("success", False),
            golden_patch=golden_patch,
            llm_usage=llm_usage,
            tool_stats=tool_stats,
            total_turns=data.get("total_turns", 0),
            select_reason=data.get("select_reason", ""),
            error=data.get("error"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert SelectorResult to dictionary."""
        return {
            "instance_id": self.instance_id,
            "generator_id": self.generator_id,
            "image": self.image,
            "success": self.success,
            "golden_patch": {
                "patch_content": self.golden_patch.patch_content,
                "test_status": self.golden_patch.test_status,
                "reasoning": self.golden_patch.reasoning,
            },
            "llm_usage": {
                "prompt_tokens": self.llm_usage.prompt_tokens,
                "completion_tokens": self.llm_usage.completion_tokens,
                "total_tokens": self.llm_usage.total_tokens,
            },
            "tool_stats": {
                BASH_TOOL_NAME: self.tool_stats.bash,
                STR_REPLACE_BASED_EDIT_TOOL_NAME: self.tool_stats.edit,
                SEARCH_TOOL_NAME: self.tool_stats.search,
                SUBMIT_RESULT_TOOL_NAME: self.tool_stats.submit_result,
            },
            "total_turns": self.total_turns,
            "select_reason": self.select_reason,
            "error": self.error,
        }
