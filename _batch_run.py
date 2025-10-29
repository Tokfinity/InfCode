import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import signal
import sys
import time
import yaml
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import multiprocessing
from functools import partial


class Config(BaseModel):
    issue_list: Optional[List[str]] = None
    pass


class Args:
    config: str
    output: str
    parallel: int
    name: str
    issue_list: str
    custom: str
    clean: bool
    dry_run: bool
    pass


class Context:
    config: Config
    args: Args

    def __init__(self, config: Config, args: Args):
        self.args = args
        self.config = config


def main():
    (args, cfg) = parse_args()
    ctx = Context(config=cfg, args=args)
    print(f"Input args: {args} {args.config}")

    out_dir = Path(ctx.args.output).joinpath(format(f"batch-{ctx.args.name}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir}is not a directory")
    if ctx.args.clean:
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()
    p = BatchProcessExecutor(ctx, out_dir=out_dir)
    p.execute_all_tasks()
    duration = int(time.time() - now)
    print(
        f"DONE Total time cost {int(duration/3600)}h {int(duration/60%60)}m {int(duration%60)}s"
    )


@dataclass
class ProcResult:
    issue: str
    idx: int
    duration: int
    summary: dict[str, Any]


def worder_func_for_gen_result(ctx: Context, out_dir: Path):
    print(f"worder_func_for_gen_result...")
    import _run
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["workspace"]["path"] = str(out_dir)
    cfg["result"]["preds"]["result"] = "result"
    cfg["runner"]["selector_result_dump_path"] = str(
        out_dir.joinpath("selector_result_dump")
    )

    start_time = time.time()

    if not ctx.args.dry_run:
        _run.gen_result(cfg)
        time.sleep(1)
    else:
        time.sleep(2)

    duration = time.time() - start_time
    print(
        f"worder_func_for_gen_result DONE time cost:{int(duration/60)}m {int(duration%60)}s"
    )


def worker_function(ctx: Context, issue: str, idx: int, out_dir: Path) -> ProcResult:
    print(f"worker_function idx:{idx} issue:{issue}")

    issue_out_dir = out_dir.joinpath("issues").joinpath(f"{idx:03d}-{issue}")
    issue_out_dir.mkdir(parents=True, exist_ok=True)

    issue_out_log_dir = issue_out_dir.joinpath("logs")
    issue_out_log_dir.mkdir(parents=True, exist_ok=True)

    generator_result_dump_path = out_dir.joinpath("generator_result_dump")
    generator_result_dump_path.mkdir(parents=True, exist_ok=True)

    selector_result_dump_path = out_dir.joinpath("selector_result_dump")
    selector_result_dump_path.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(issue_out_dir.joinpath("stdout.log"), "a")
    sys.stderr = open(issue_out_dir.joinpath("stdout.log"), "a")
    # signal.signal(signal.SIGINT, signal.SIG_IGN)

    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    start_time = time.time()

    import _run

    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["log"]["base_path"] = str(issue_out_log_dir)
    cfg["runner"]["generator_result_dump_path"] = str(generator_result_dump_path)
    cfg["runner"]["selector_result_dump_path"] = str(selector_result_dump_path)
    if not ctx.args.dry_run:
        summary = _run.run_one_issue(cfg, issue=issue)
        time.sleep(1)
    else:
        time.sleep(2)
        summary = {}
        summary["success"] = 1
        summary["failed"] = 0
        summary["total"] = 1

    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    duration = time.time() - start_time
    print(
        f"worker_function DONE idx:{idx} issue:{issue} 耗时:{int(duration/60)}m {int(duration%60)}s"
    )
    if summary["success"] > 0:
        add_done_set(out_dir, issue)

    return ProcResult(duration=duration, issue=issue, idx=idx, summary=summary)


class BatchProcessExecutor:
    ctx: Context
    out_dir: Path

    def __init__(self, ctx: Context, out_dir: Path):
        self.ctx = ctx
        self.out_dir = out_dir

    def execute_all_tasks(self):
        done_set = load_done_set(self.out_dir)
        parallel = self.ctx.args.parallel
        formatted_issues = []
        for idx, issue in enumerate(self.ctx.config.issue_list, 1):
            if issue in done_set:
                print(f"done set: skip {idx}, {issue}")
            else:
                formatted_issues.append((issue, idx, self.out_dir))
        with multiprocessing.Pool(processes=parallel, maxtasksperchild=1) as pool:
            try:
                worker = partial(worker_function, self.ctx)
                results = pool.starmap(worker, formatted_issues)
            except KeyboardInterrupt as e:
                print(f"ctrl-c received, exit")
                pool.terminate()
        cum_total = 0
        cum_success = 0
        cum_failed = 0
        for result in results:
            total = result.summary["total"]
            success = result.summary["success"]
            failed = result.summary["failed"]
            cum_total += total
            cum_success += success
            cum_failed += failed

        print(f"Total instances: {cum_total}")
        print(f"Success: {cum_success}")
        print(f"Fail: {cum_failed}")
        print(f"Success rate: {(cum_success/cum_total*100):.1f}%" if cum_total > 0 else "0%")

        print(f"Start generating final results")
        process = multiprocessing.Process(
            target=worder_func_for_gen_result, args=(self.ctx, self.out_dir)
        )
        process.start()
        process.join()


def parse_args() -> tuple[Args, Config]:
    parser = argparse.ArgumentParser(description="This is a concurrent execution tool.")
    parser.add_argument("-c", "--config", help="config file", required=True)
    parser.add_argument(
        "--name", help="task name,which will be concatenated as a directory name under the output path.", required=True
    )
    parser.add_argument("-i", "--issue-list", help="question list", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="output directory, ./batch_out/as default",
        type=str,
        default="./batch_out/",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        help="Parallelism，20 as default",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--clean", help="Clean up data with the same name in the output directory before starting.", action="store_true"
    )
    parser.add_argument(
        "--dry-run", help="Skip the actual inference task execution, but proceed with all other logic.", action="store_true"
    )

    args: Args = parser.parse_args()
    setattr(args, "custom", "str")

    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    cfg = Config(**config_data)

    if args.issue_list:
        issues: list[str] = []
        with open(args.issue_list, "r", encoding="utf-8") as f:
            for line in f:
                line_clean = line.strip()
                if line_clean:
                    issues.append(str(line_clean))
        cfg.issue_list = issues

    print(f"list len = {len(cfg.issue_list)}")
    return (args, cfg)


def signal_handler(signum, frame):
    print(f"Received signal {signum}, terminating child processes...")
    # sys.exit(0)


def load_done_set(out_dir: Path) -> set[str]:
    file_name = out_dir.joinpath("done.txt")
    if not file_name.exists():
        return set()
    with open(file_name, "r") as f:
        return set(line.strip() for line in f if line.strip())


def add_done_set(out_dir: Path, issue: str):
    file_name = out_dir.joinpath("done.txt")
    with open(file_name, "a") as f:
        f.write(f"{issue}\n")


if __name__ == "__main__":
    main()
