from typing import List, Dict, Any
from pathlib import Path
import json
import yaml
import asyncio
from datetime import datetime
from traceback import format_exc
from src.tools.executor import Executor
from src.tools import BashTool, TextEditorTool, SearchTool, SubmitResultTool
from src.managers.result_builder.result_builder import ResultBuilder
from src.tools.base import (
    ToolExecutor,
    BASH_TOOL_NAME,
    STR_REPLACE_BASED_EDIT_TOOL_NAME,
    SEARCH_TOOL_NAME,
    SUBMIT_RESULT_TOOL_NAME,
)
from src.managers.log.logger import create_logger, Logger
from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.image_builder.build_image import SWEBenchImageBuilder
from src.managers.prompts.prompts_manager import PromptsManager
from src.managers.loop.patch_generator import PatchGenerator
from src.managers.loop.patch_selector import PatchSelector
from src.managers.loop.types import GeneratorResult, SelectorResult


class SelectorLoop:
    def __init__(
        self,
        instance_id: str,
        image_name: str,
        runner_log_base: Path,
        llm_manager: LLMAPIManager | None,
        prompts_manager: PromptsManager | None,
        instance_data: Dict[str, Any],
        config: Dict[str, Any],
    ):
        self.instance_id = instance_id
        self.image_name = image_name
        self.llm_manager = llm_manager
        self.prompts_manager = prompts_manager
        self.instance_data = instance_data
        self.config = config
        self.log_dir = runner_log_base / "run" / instance_id / "selector"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(
            log_base_path=str(self.log_dir.parent),
            logger_name=f"selector_{instance_id}",
            console_output=True,
            instance_id=self.log_dir.name,
        )

    def _dump_select_result(self, result: SelectorResult) -> None:
        try:
            runner_cfg = (
                self.config.get("runner", {}) if isinstance(self.config, dict) else {}
            )
            dump_dir_str = runner_cfg.get(
                "selector_result_dump_path", "workspace/selector_result_dump"
            )
            dump_dir = Path(dump_dir_str)
            dump_dir.mkdir(parents=True, exist_ok=True)

            out_path = dump_dir / f"{self.instance_id}.json"

            payload = (
                result.to_dict()
                if hasattr(result, "to_dict") and callable(getattr(result, "to_dict"))
                else {}
            )

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.warning(
                f"dump selector result fail: {e}, traceback: {format_exc()}"
            )

    def _load_dumped_generator_results(self) -> List[GeneratorResult]:
        try:
            runner_cfg = (
                self.config.get("runner", {}) if isinstance(self.config, dict) else {}
            )
            dump_dir_str = runner_cfg.get(
                "generator_result_dump_path", "workspace/generator_result_dump"
            )
            dump_path = Path(dump_dir_str) / f"{self.instance_id}.json"
            if not dump_path.exists():
                self.logger.warning(f"fail to find dump 文件: {dump_path}")
                return []
            with open(dump_path, "r", encoding="utf-8") as f:
                data = json.load(f) or []
            results: List[GeneratorResult] = []
            for item in data:
                try:
                    if isinstance(item, dict):
                        results.append(GeneratorResult.from_dict(item))
                except Exception as e:
                    self.logger.warning(f"parse dump fail: {e}")
                    continue
            return results
        except Exception as e:
            self.logger.warning(f"load dump fail: {e}, traceback: {format_exc()}")
            return []

    async def select(self, generator_results: List[GeneratorResult]) -> SelectorResult:

        if bool(self.config.get("runner", {}).get("skip_generator", False)):
            self.logger.info("jump generator，selector will load generator results from dump files")
            generator_results = self._load_dumped_generator_results()
            self.logger.info(f"load from dump: {len(generator_results)} candidates")

        if not generator_results:
            from src.managers.loop.types import (
                SelectorResult,
                PatchInfo,
                LLMUsage,
                ToolStats,
            )

            self.logger.error("No choosable candidates found")
            return SelectorResult(
                instance_id=self.instance_id,
                generator_id=-1,
                image="",
                success=False,
                golden_patch=PatchInfo(patch_content="", test_status="", reasoning=""),
                llm_usage=LLMUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
                tool_stats=ToolStats(bash=0, edit=0, search=0, submit_result=0),
                total_turns=0,
                select_reason="",
                error="No candidates available",
            )

        self.logger.info(f"Start choosing best result，{len(generator_results)} candidates in total")
        executor = Executor(self.image_name, self.logger)
        bash_tool = BashTool(
            model_provider=None,
            executor=executor,
            logger=self.logger,
            config=self.config,
        )
        edit_tool = TextEditorTool(
            model_provider=None,
            executor=executor,
            logger=self.logger,
            config=self.config,
        )
        search_tool = SearchTool(
            model_provider=None,
            executor=executor,
            logger=self.logger,
            config=self.config,
        )
        tool_executor = ToolExecutor([bash_tool, edit_tool, search_tool], self.logger)

        code, out = executor.execute("0", "echo READY && rg --version || true")
        self.logger.info(f"Container Health check: exit={code}, out=\n{out}")

        successful_candidates = [r for r in generator_results if r.success]

        if not successful_candidates:
            self.logger.warning("No successful candidates found, randomly choose one from all candidates")
            candidates = generator_results
        else:
            self.logger.info(f"Find {len(successful_candidates)} successful candidates")
            candidates = successful_candidates

        patch_selector = PatchSelector(
            instance_id=self.instance_id,
            instance_data=self.instance_data,
            logger=self.logger,
            prompts_manager=self.prompts_manager,
            llm_manager=self.llm_manager,
            tool_executor=tool_executor,
            config=self.config,
        )

        selected = await patch_selector._select_patch(candidates=candidates)

        try:
            runner_cfg = (
                self.config.get("runner", {}) if isinstance(self.config, dict) else {}
            )
            if bool(runner_cfg.get("selector_result_dump", False)):
                self._dump_select_result(selected)
        except Exception as e:
            self.logger.warning(
                f"Error occurred when choosing dump : {e}, traceback: {format_exc()}"
            )

        # import random
        # selected = random.choice(candidates)

        self.logger.info(f"Choosing complete: choose#{selected.generator_id}.")

        return selected


class GeneratorLoop:
    def __init__(
        self,
        instance_id: str,
        image_name: str,
        runner_log_base: Path,
        llm_manager: LLMAPIManager | None,
        prompts_manager: PromptsManager | None,
        instance_data: Dict[str, Any],
        config: Dict[str, Any],
        generator_id: int = 0,
    ):
        self.instance_id = instance_id
        self.image_name = image_name
        self.generator_id = generator_id
        self.llm_manager = llm_manager
        self.prompts_manager = prompts_manager
        self.instance_data = instance_data
        self.config = config
        self.log_dir = (
            runner_log_base / "run" / instance_id / "generator" / f"{generator_id:03d}"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(
            log_base_path=str(self.log_dir.parent),
            logger_name=f"generator_{instance_id}_{generator_id:03d}",
            console_output=True,
            instance_id=self.log_dir.name,
        )

    async def generate(self) -> GeneratorResult:
        executor: Executor | None = None
        try:
            self.logger.info(
                f"Activate instance GeneratorLoop #{self.generator_id:03d}: {self.instance_id} -> {self.image_name}"
            )
            self.logger.info(f"Use image: {self.image_name}")
            executor = Executor(self.image_name, self.logger)

            bash_tool = BashTool(
                model_provider=None,
                executor=executor,
                logger=self.logger,
                config=self.config,
            )
            edit_tool = TextEditorTool(
                model_provider=None,
                executor=executor,
                logger=self.logger,
                config=self.config,
            )
            search_tool = SearchTool(
                model_provider=None,
                executor=executor,
                logger=self.logger,
                config=self.config,
            )
            submit_result_tool = SubmitResultTool(
                model_provider=None,
                executor=executor,
                logger=self.logger,
                config=self.config,
            )
            tool_executor = ToolExecutor(
                [bash_tool, edit_tool, search_tool, submit_result_tool], self.logger
            )
            # tool_executor = ToolExecutor([bash_tool, edit_tool])

            # optional: do a container health check
            code, out = executor.execute("0", "echo READY && rg --version || true")
            self.logger.info(f"Container Health Check: exit={code}, out=\n{out}")


            patch_generator = PatchGenerator(
                instance_id=self.instance_id,
                instance_data=self.instance_data,
                logger=self.logger,
                prompts_manager=self.prompts_manager,
                llm_manager=self.llm_manager,
                tool_executor=tool_executor,
                config=self.config,
            )

            patch_result = await patch_generator._generate_patch()

            if patch_result is None:
                result_data = {
                    "instance_id": self.instance_id,
                    "generator_id": self.generator_id,
                    "image": self.image_name,
                    "success": False,
                    "golden_patch": [],
                    "llm_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "tool_stats": {
                        BASH_TOOL_NAME: 0,
                        STR_REPLACE_BASED_EDIT_TOOL_NAME: 0,
                        SEARCH_TOOL_NAME: 0,
                        SUBMIT_RESULT_TOOL_NAME: 0,
                    },
                    "total_turns": 0,
                }
            else:
                result_data = {
                    "instance_id": self.instance_id,
                    "generator_id": self.generator_id,
                    "image": self.image_name,
                    "success": patch_result.get("success", False),
                    "golden_patch": patch_result.get("golden_patch", []),
                    "llm_usage": patch_result.get(
                        "llm_usage",
                        {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    ),
                    "tool_stats": patch_result.get(
                        "tool_stats",
                        {
                            BASH_TOOL_NAME: 0,
                            STR_REPLACE_BASED_EDIT_TOOL_NAME: 0,
                            SEARCH_TOOL_NAME: 0,
                            SUBMIT_RESULT_TOOL_NAME: 0,
                        },
                    ),
                    "total_turns": patch_result.get("total_turns", 0),
                }
            self.logger.debug(f"[Generator Loop] result_data: {result_data}")
            return GeneratorResult.from_dict(result_data)
        except Exception as e:
            self.logger.error(
                f"Instance {self.instance_id} Generator #{self.generator_id:03d} fail: {e}, traceback: {format_exc()}"
            )
            error_data = {
                "instance_id": self.instance_id,
                "generator_id": self.generator_id,
                "image": self.image_name,
                "success": False,
                "error": str(e),
                "golden_patch": [],
                "llm_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "tool_stats": {
                    BASH_TOOL_NAME: 0,
                    STR_REPLACE_BASED_EDIT_TOOL_NAME: 0,
                    SEARCH_TOOL_NAME: 0,
                    SUBMIT_RESULT_TOOL_NAME: 0,
                },
                "total_turns": 0,
            }
            return GeneratorResult.from_dict(error_data)
        finally:
            if executor:
                try:
                    executor.shutdown()
                except Exception:
                    pass


class Runner:
    def __init__(self, cfg: Dict[str, Any], instance_ids: List[str] = None):
        self.cfg = cfg
        dataset_cfg = cfg.get("dataset", {})
        workspace_cfg = cfg.get("workspace", {})
        builder_cfg = cfg.get("builder", {})
        log_cfg = cfg.get("log", {})
        runner_cfg = cfg.get("runner", {})
        providers_cfg = cfg.get("providers", {})
        self.instance_ids = instance_ids
        self.dataset_name = dataset_cfg.get("name", "princeton-nlp/SWE-bench_Lite")
        self.dataset_split = dataset_cfg.get("split", "dev")
        self.max_workers = int(builder_cfg.get("max_workers", 2))
        self.generator_loop_concurrency = int(
            runner_cfg.get("generator_concurrency", 2)
        )

        log_base_path = log_cfg.get("base_path", "workspace/logs")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.logs_base = Path(log_base_path) / timestamp
        self.logs_base.mkdir(parents=True, exist_ok=True)

        self.logger = Logger(
            log_base_path=str(self.logs_base.parent),
            logger_name="main",
            console_output=True,
            instance_id=self.logs_base.name,
        )

        self.builder = self.load_images()
        self.logger.debug(f"builder instance_to_image: {self.builder.instance_to_image}.")
        self.llm_manager = LLMAPIManager(logger=self.logger, config=cfg)

        self.prompts_manager: PromptsManager | None = None
        try:
            self.prompts_manager = PromptsManager(cfg)
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize PromptsManager: {e}, traceback: {format_exc()}"
            )
            self.prompts_manager = None

    def dump_generator_results(
        self, instance_id: str, generator_results: List[GeneratorResult]
    ) -> None:
        """Dump generator results to disk if enabled in config.

        - Reads runner.generator_result_dump (bool) and runner.generator_result_dump_path (str)
        - When enabled, writes JSON file named by instance_id under the dump path
        """
        try:
            runner_cfg: Dict[str, Any] = (
                self.cfg.get("runner", {}) if isinstance(self.cfg, dict) else {}
            )
            enabled: bool = bool(runner_cfg.get("generator_result_dump", False))
            if not enabled:
                return

            dump_dir_str: str = runner_cfg.get(
                "generator_result_dump_path", "workspace/generator_result_dump"
            )
            dump_dir = Path(dump_dir_str)
            dump_dir.mkdir(parents=True, exist_ok=True)

            out_path = dump_dir / f"{instance_id}.json"

            # Convert results to serializable form
            serialized: List[Dict[str, Any]] = []
            for r in generator_results:
                try:
                    if hasattr(r, "to_dict") and callable(getattr(r, "to_dict")):
                        serialized.append(r.to_dict())
                    elif isinstance(r, dict):
                        serialized.append(r)
                    else:
                        # Fallback: best-effort representation
                        serialized.append({"repr": repr(r)})
                except Exception as e:  # noqa: BLE001
                    serialized.append({"error": f"failed to serialize item: {e}"})

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)

            self.logger.info(
                f"Generator results write to: {out_path} ({len(serialized)} items in total)"
            )
        except Exception as e:
            self.logger.warning(
                f"dump_generator_results failed: {e}, traceback: {format_exc()}"
            )

    def load_images(self):
        self.logger.info("Initialize SWEBenchImageBuilder and ready to load image...")
        builder = SWEBenchImageBuilder(
            dataset_name=self.dataset_name,
            split=self.dataset_split,
            logger=self.logger,
        )
        builder.load_images(instance_ids=self.instance_ids) 
        return builder

    async def _run_one(
        self,
        instance_id: str,
        image_name: str,
        instance_data: Dict[str, Any],
        generator_id: int = 0,
    ) -> GeneratorResult:
        loop = GeneratorLoop(
            instance_id,
            image_name,
            self.logs_base,
            self.llm_manager,
            self.prompts_manager,
            instance_data,
            self.cfg,
            generator_id,
        )
        return await loop.generate()

    async def process_one_instance(self, instance: Dict[str, Any]) -> SelectorResult:
        instance_id = instance["instance_id"]
        try:
            image_name = self.builder.get_image_name(instance_id)
        except KeyError:
            self.logger.warning(f"Jump instance (image mapping unfind): {instance_id}")
            return None

        self.logger.info(f"Start processing instance: {instance_id}")

        # optional: jump generator, let selector load from dump directly and choose
        skip_generator = bool(self.cfg.get("runner", {}).get("skip_generator", False))
        if skip_generator:
            self.logger.info(
                "Jump GeneratorLoop，Selector load from dump directly"
            )
            selector = SelectorLoop(
                instance_id=instance_id,
                image_name=image_name,
                runner_log_base=self.logs_base,
                llm_manager=self.llm_manager,
                prompts_manager=self.prompts_manager,
                instance_data=instance,
                config=self.cfg,
            )
            selected_result = await selector.select(
                []
            )
            self.logger.info(
                f"Instance {instance_id} done(generating jumped)，Generator #{selected_result.generator_id:03d} chosen"
            )
            return selected_result

        generator_load_dump = bool(
            self.cfg.get("runner", {}).get("generator_load_dump_result", False)
        )
        valid_results = []

        if generator_load_dump:
            runner_cfg = self.cfg.get("runner", {})
            dump_path = runner_cfg.get(
                "generator_result_dump_path", "workspace/generator_result_dump"
            )

            if self._check_dump_result(dump_path, instance_id):
                self.logger.info(f"load generator results from dump: {instance_id}")
                valid_results = self._load_generator_dump_result(dump_path, instance_id)
            else:
                self.logger.info(
                    f"Fail to find generator dump file，Generate candidate patches concurrently: {instance_id}"
                )

        if not valid_results:
            generator_tasks = []
            for generator_id in range(self.generator_loop_concurrency):
                task = asyncio.create_task(
                    self._run_one(instance_id, image_name, instance, generator_id)
                )
                generator_tasks.append(task)

            generator_results = await asyncio.gather(
                *generator_tasks, return_exceptions=True
            )
            self.logger.debug(
                f"In process_one_instance, generator_results len: {len(generator_results)}"
            )

            for result in generator_results:
                if isinstance(result, Exception):
                    self.logger.error(f"GeneratorLoop exception: {result}")
                else:
                    valid_results.append(result)
            self.logger.debug(
                f"In process_one_instance, valid_results len: {len(valid_results)}"
            )
            # optional: Dump the generator results for subsequent selector debugging.
            try:
                self.dump_generator_results(instance_id, valid_results)
            except Exception:
                # Dump failure should not block the main process/flow.
                pass

        if not valid_results:
            self.logger.warning(f"Instance {instance_id} has no valid GeneratorLoop results")
            return None

        selector_load_dump = bool(
            self.cfg.get("runner", {}).get("selector_load_dump_result", False)
        )

        if selector_load_dump:
            runner_cfg = self.cfg.get("runner", {})
            dump_path = runner_cfg.get(
                "selector_result_dump_path", "workspace/selector_result_dump"
            )

            if self._check_dump_result(dump_path, instance_id):
                self.logger.info(f"load selector results from dump file: {instance_id}")
                try:
                    dump_dir = Path(dump_path)
                    file_path = dump_dir / f"{instance_id}.json"
                    with open(file_path, "r", encoding="utf-8") as f:
                        selected_data = json.load(f)

                    from src.managers.loop.types import SelectorResult

                    selected_result = SelectorResult.from_dict(selected_data)

                    self.logger.info(
                        f"Instance{instance_id} process done (load from dump)，Generator #{selected_result.generator_id:03d} chosen"
                    )
                    return selected_result
                except Exception as e:
                    self.logger.warning(
                        f"Fail to load selector dump result: {e}, execute normal choosing procedure"
                    )
            else:
                self.logger.info(
                    f"Fail to load selector dump result, execute normal choosing procedure: {instance_id}"
                )

        self.logger.info(f"Star choosing best result for instance {instance_id}")
        selector = SelectorLoop(
            instance_id=instance_id,
            image_name=image_name,
            runner_log_base=self.logs_base,
            llm_manager=self.llm_manager,
            prompts_manager=self.prompts_manager,
            instance_data=instance,
            config=self.cfg,
        )
        selected_result = await selector.select(valid_results)

        self.logger.info(
            f"Instance {instance_id} processed，Generator #{selected_result.generator_id:03d} chosen"
        )
        return selected_result

    async def run(self) -> Dict[str, Any]:

        assert self.builder is not None

        if self.instance_ids:
            target_ids = set(self.instance_ids)
            instances_to_run = [
                inst
                for inst in self.builder.full_dataset
                if inst.get("instance_id") in target_ids
            ]
        else:
            instances_to_run = list(self.builder.full_dataset)

        self.logger.info(f"Start to process {len(instances_to_run)} instances")

        final_results = []
        for i, instance in enumerate(instances_to_run, 1):
            self.logger.info(
                f"process instance{i}/{len(instances_to_run)}: {instance['instance_id']}"
            )
            try:
                result = await self.process_one_instance(instance)
                if result is not None:
                    final_results.append(result)
            except Exception as e:
                self.logger.error(
                    f"Instance {instance['instance_id']} process fail: {e}, traceback: {format_exc()}."
                )
                final_results.append(
                    SelectorResult(
                        instance_id=instance["instance_id"],
                        generator_id=0,
                        success=False,
                        golden_patch=None,
                    )
                )

        summary = self._calculate_summary(final_results)

        self.logger.info(
            f"Done. Total={summary['total']} Success={summary['success']} Fail={summary['failed']}"
        )
        return summary

    def _check_dump_result(self, dump_path: str, instance_id: str) -> bool:
        try:
            dump_dir = Path(dump_path)
            file_path = dump_dir / f"{instance_id}.json"
            return file_path.exists()
        except Exception as e:
            self.logger.warning(f"Fail to check dump file: {e}")
            return False

    def _load_generator_dump_result(
        self, dump_path: str, instance_id: str
    ) -> List[GeneratorResult]:
        try:
            dump_dir = Path(dump_path)
            file_path = dump_dir / f"{instance_id}.json"

            if not file_path.exists():
                self.logger.warning(f"Generator dump file does not exist: {file_path}")
                return []

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f) or []

            results: List[GeneratorResult] = []
            for item in data:
                try:
                    if isinstance(item, dict):
                        results.append(GeneratorResult.from_dict(item))
                    else:
                        self.logger.warning(f"Jump non-dict dump: {type(item)}")
                except Exception as e:
                    self.logger.warning(f"parse dump fail: {e}")
                    continue

            self.logger.info(f"load  {len(results)} GeneratorResults from dump")
            return results
        except Exception as e:
            self.logger.warning(
                f"Load generator dump results failed: {e}, traceback: {format_exc()}"
            )
            return []

    def _calculate_summary(self, results: List[SelectorResult]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total": 0,
            "success": 0,
            "failed": 0,
        }

        for r in results:
            summary["total"] += 1
            if r.success:
                summary["success"] += 1
            else:
                summary["failed"] += 1

        return summary


def run_one_issue(cfg, issue: str) -> dict[str, Any]:
    ids = [issue]
    summary = {}
    if not cfg.get("runner", {}).get("skip_selector", False):
        runner = Runner(cfg, ids)
        summary = asyncio.run(runner.run())
        print("\n" + "=" * 80)
        print("total results")
        print("=" * 80)
        print(f"Total instances: {summary['total']}")
        print(f"Success: {summary['success']}")
        print(f"Fail: {summary['failed']}")
        print(
            f"Success rate: {(summary['success']/summary['total']*100):.1f}%"
            if summary["total"] > 0
            else "0%"
        )

    return summary


def gen_result(cfg):
    result_builder = ResultBuilder(cfg)
    result_path = result_builder.build_preds()


def main() -> None:
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    test_instance_ids = ["astropy__astropy-14309"]
    if not cfg.get("runner", {}).get("skip_selector", False):
        runner = Runner(cfg, test_instance_ids)
        summary = asyncio.run(runner.run())
        print("\n" + "=" * 80)
        print("Total results")
        print("=" * 80)
        print(f"Total instances: {summary['total']}")
        print(f"Success: {summary['success']}")
        print(f"Fail: {summary['failed']}")
        print(
            f"Success rate: {(summary['success']/summary['total']*100):.1f}%"
            if summary["total"] > 0
            else "0%"
        )

    result_builder = ResultBuilder(cfg)
    result_builder.build_preds()


if __name__ == "__main__":
    main()
