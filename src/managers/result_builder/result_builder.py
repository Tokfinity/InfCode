from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json


class ResultBuilder:
    """
    Build preds.json

    Iterate through JSON files named by instance_id in the directory specified by `runner.selector_result_dump_path` in the configuration.
    - Parse the `golden_patch` field from each JSON file and extract the patch text as `model_patch`.
    - Read the first top-level field from providers and the first model under it, and concatenate them to form `model_name_or_path`.
    - The output location is `{workspace.path}/{result.preds.path}/preds.json`.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    def _get_selector_dump_dir(self) -> Path:
        runner_cfg = self.config.get("runner", {}) if isinstance(self.config, dict) else {}
        dump_dir_str = runner_cfg.get(
            "selector_result_dump_path", "workspace/selector_result_dump"
        )
        return Path(dump_dir_str)

    def _get_preds_output_dir(self) -> Path:
        workspace_cfg = self.config.get("workspace", {}) if isinstance(self.config, dict) else {}
        result_cfg = self.config.get("result", {}) if isinstance(self.config, dict) else {}
        preds_cfg = result_cfg.get("preds", {}) if isinstance(result_cfg, dict) else {}

        workspace_path = workspace_cfg.get("path", "workspace")
        preds_path = preds_cfg.get("path", "result")
        return Path(workspace_path) / preds_path

    def _get_model_name_or_path(self) -> str:
        providers = self.config.get("providers", {}) if isinstance(self.config, dict) else {}
        if not isinstance(providers, dict) or not providers:
            return ""
        first_provider_name = next(iter(providers.keys()))
        first_models = providers.get(first_provider_name, [])
        if isinstance(first_models, list) and first_models:
            first_model = first_models[0]
        else:
            first_model = ""
        return f"{first_provider_name}/{first_model}" if first_provider_name and first_model else ""

    @staticmethod
    def _extract_model_patch(golden_patch: Any) -> str:
        """
        Extract patch content from golden_patch

        Forms supported:
        - dict: prioritize extract 'patch_content', then attempt `model_patch`
        - string: Directly return
        - other: return empty string
        """
        if isinstance(golden_patch, dict):
            if "patch_content" in golden_patch and isinstance(golden_patch["patch_content"], str):
                return golden_patch["patch_content"]
            if "model_patch" in golden_patch and isinstance(golden_patch["model_patch"], str):
                return golden_patch["model_patch"]
            return ""
        if isinstance(golden_patch, str):
            return golden_patch
        return ""

    def build_preds(self) -> Path:
        dump_dir = self._get_selector_dump_dir()
        output_dir = self._get_preds_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "preds.json"

        model_name_or_path = self._get_model_name_or_path()

        # SWE-bench evaluation expects: list[dict], each element includes instance_id / model_patch / model
        predictions: list[dict[str, str]] = []

        if dump_dir.exists() and dump_dir.is_dir():
            for path in sorted(dump_dir.glob("*.json")):
                try:
                    instance_id = path.stem
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    golden_patch = data.get("golden_patch", {}) if isinstance(data, dict) else {}
                    model_patch = self._extract_model_patch(golden_patch)
                    predictions.append(
                        {
                            "instance_id": instance_id,
                            "model_patch": model_patch,
                            "model_name_or_path": model_name_or_path,
                        }
                    )
                except Exception:
                    continue

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        return output_file


