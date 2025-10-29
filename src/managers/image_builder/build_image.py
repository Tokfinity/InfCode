"""
SWE_bench Loader
Load the dataset and clone the repo to the workspace
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import io
import json
import subprocess
import shutil
import time
import atexit
import tempfile
import re
from pathlib import Path
from typing import List, NamedTuple, Optional, Any, Union
from datasets import load_dataset
from traceback import format_exc
import docker
from src.managers.image_builder.dockerfiles import _DOCKERFILE_USER_IMAGE_PY
from src.managers.log.logger import Logger as CustomLogger
from src.managers.image_builder.print_redirect import redirect_swebench_prints
from src.managers.image_builder.logger_patch import patched_build_instance_images

# Always use temporary directory for logging (no persistent logs)
temp_dir = tempfile.mkdtemp(prefix="swebench_temp_")
image_builder_path = temp_dir


def cleanup_temp_dir():
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass


atexit.register(cleanup_temp_dir)

# Override SWE-bench constants BEFORE importing any SWE-bench modules
import swebench.harness.constants
import logging

# Set all log directories to temporary locations BEFORE any imports
swebench.harness.constants.BASE_IMAGE_BUILD_DIR = (
    Path(temp_dir) / "build_images" / "base"
)
swebench.harness.constants.ENV_IMAGE_BUILD_DIR = Path(temp_dir) / "build_images" / "env"
swebench.harness.constants.INSTANCE_IMAGE_BUILD_DIR = (
    Path(temp_dir) / "build_images" / "instances"
)
swebench.harness.constants.RUN_EVALUATION_LOG_DIR = Path(temp_dir) / "run_evaluation"
swebench.harness.constants.RUN_VALIDATION_LOG_DIR = Path(temp_dir) / "run_validation"

Path(temp_dir, "build_images", "base").mkdir(parents=True, exist_ok=True)
Path(temp_dir, "build_images", "env").mkdir(parents=True, exist_ok=True)
Path(temp_dir, "build_images", "instances").mkdir(parents=True, exist_ok=True)
Path(temp_dir, "run_evaluation").mkdir(parents=True, exist_ok=True)
Path(temp_dir, "run_validation").mkdir(parents=True, exist_ok=True)


from swebench.harness.docker_build import TestSpec, build_instance_images
from swebench.harness.utils import SWEbenchInstance, load_swebench_dataset
from swebench.harness.test_spec.test_spec import make_test_spec


# Create a no-op logger
def setup_logger_noop(
    instance_id: str, log_file: Path, mode="w", add_stdout: bool = False
):
    """No-op logger that doesn't create any files or write any logs"""
    logger = logging.getLogger(f"{instance_id}.noop")
    # Set level to CRITICAL to suppress all messages
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    handler = logging.NullHandler()
    logger.addHandler(handler)
    setattr(logger, "log_file", None)  # No log file
    return logger


# Replace the original setup_logger function
import swebench.harness.docker_build

swebench.harness.docker_build.setup_logger = setup_logger_noop


# Override the close_logger function to be a no-op
def close_logger_noop(logger):
    """No-op close function"""
    pass


swebench.harness.docker_build.close_logger = close_logger_noop

# Override ALL the build directory constants in the docker_build module
swebench.harness.docker_build.BASE_IMAGE_BUILD_DIR = (
    Path(temp_dir) / "build_images" / "base"
)
swebench.harness.docker_build.ENV_IMAGE_BUILD_DIR = (
    Path(temp_dir) / "build_images" / "env"
)
swebench.harness.docker_build.INSTANCE_IMAGE_BUILD_DIR = (
    Path(temp_dir) / "build_images" / "instances"
)

swebench.harness.constants.BASE_IMAGE_BUILD_DIR = (
    Path(temp_dir) / "build_images" / "base"
)
swebench.harness.constants.ENV_IMAGE_BUILD_DIR = Path(temp_dir) / "build_images" / "env"
swebench.harness.constants.INSTANCE_IMAGE_BUILD_DIR = (
    Path(temp_dir) / "build_images" / "instances"
)

original_build_image = swebench.harness.docker_build.build_image


def patched_build_image(
    image_name, setup_scripts, dockerfile, platform, client, build_dir, nocache=False
):
    """Patched build_image function that ensures directories exist"""
    build_dir.mkdir(parents=True, exist_ok=True)

    return original_build_image(
        image_name, setup_scripts, dockerfile, platform, client, build_dir, nocache
    )


swebench.harness.docker_build.build_image = patched_build_image


class SWEBenchLoader:
    """SWE_bench dataset loader"""

    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        split_name: str = "dev",
        workspace_path: str = "workspace",
        logger: Optional[Union[logging.Logger, CustomLogger]] = None,
    ):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.workspace_path = Path(workspace_path)
        self.logger = logger
        self.dataset = None

        self.workspace_path.mkdir(parents=True, exist_ok=True)

        if self.logger:
            self.logger.info(
                f"Initialize SWE-bench DataLoader - Dataset: {dataset_name}, Split: {split_name}, Workspace: {workspace_path}"
            )

    def _convert_repo_to_url(self, repo_name: str) -> str:
        """
        Convert repo name to GitHub URL

        Args:
            repo_name: repo name such as 'owner/repo'

        Returns:
            str: Full github URL
        """
        if repo_name.startswith("http"):
            return repo_name
        else:
            return f"https://github.com/{repo_name}.git"

    def load_dataset(self):
        """
        Load complete dataset

        Returns:
            Dataset: loaded dataset
        """
        try:
            if self.logger:
                self.logger.info(
                    f"Start loading dataset: {self.dataset_name}, Split: {self.split_name}"
                )

            self.dataset = load_dataset(self.dataset_name, split=self.split_name)

            if self.logger:
                self.logger.info(f"Successfully loaded dataset，including {len(self.dataset)} records")

            return self.dataset

        except Exception as e:
            error_msg = f"Failed to load dataset: {str(e)}, traceback: {format_exc()}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def clone_repository(
        self,
        repo_url: str,
        target_path: Path,
        commit_hash: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        force_reclone: bool = True,
    ) -> bool:
        """
        Clone GitHub repository from repo_url to target_path (Retry supported)

        Args:
            repo_url: repo URL
            target_path: target path
            commit_hash: specific commit hash(optional)
            max_retries: max retries
            retry_delay: retry delay
            force_reclone: whether to force cloning(deleting existing repo)

        Returns:
            bool: whether clone succeeded
        """
        for attempt in range(max_retries + 1):
            try:
                if force_reclone and target_path.exists():
                    if self.logger:
                        self.logger.debug(f"Delete existing repo: {target_path}")
                    shutil.rmtree(target_path)

                # If the target path exists with no force_reclone, check whether it's a valid Git repo
                elif target_path.exists() and (target_path / ".git").exists():
                    if self.logger:
                        self.logger.debug(f"Repo exists, skip clone: {target_path}")

                    if commit_hash:
                        return self._checkout_commit(target_path, commit_hash)
                    return True

                if self.logger:
                    self.logger.debug(
                        f"Start cloning repo (Try {attempt + 1}/{max_retries + 1}): {repo_url} -> {target_path}"
                    )

                cmd = ["git", "clone", repo_url, str(target_path)]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )

                if result.returncode != 0:
                    error_msg = (
                        f"Git clone failed (try {attempt + 1}): {result.stderr.strip()}"
                    )
                    if self.logger:
                        self.logger.warning(error_msg)

                    if attempt < max_retries:
                        if self.logger:
                            self.logger.info(f"Retry after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        if self.logger:
                            self.logger.error(
                                f"All retries failed: {repo_url}"
                            )
                        return False

                if commit_hash:
                    if not self._checkout_commit(target_path, commit_hash):
                        if self.logger:
                            self.logger.warning(
                                f"Clone succeeded but commit fail: {commit_hash}"
                            )

                if self.logger:
                    self.logger.info(f"Clone success: {repo_url}")

                return True

            except subprocess.TimeoutExpired:
                error_msg = f"Git operation timeout(try {attempt + 1}): {repo_url}"
                if self.logger:
                    self.logger.warning(error_msg)

                if target_path.exists():
                    shutil.rmtree(target_path, ignore_errors=True)

                if attempt < max_retries:
                    if self.logger:
                        self.logger.info(f"Retry after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    if self.logger:
                        self.logger.error(f"All retries failed due to timeout: {repo_url}")
                    return False

            except Exception as e:
                error_msg = f"Clone repo error (try {attempt + 1}): {str(e)}, traceback: {format_exc()}."
                if self.logger:
                    self.logger.warning(error_msg)

                if target_path.exists():
                    shutil.rmtree(target_path, ignore_errors=True)

                if attempt < max_retries:
                    if self.logger:
                        self.logger.info(f"Retry after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    if self.logger:
                        self.logger.error(f"All retries failed: {repo_url}")
                    return False

        return False

    def _checkout_commit(self, repo_path: Path, commit_hash: str) -> bool:
        """
        Change to a specific commit.

        Args:
            repo_path: repo path
            commit_hash: commit hash

        Returns:
            bool: change succeeded
        """
        try:
            if self.logger:
                self.logger.debug(f"Changed to commit: {commit_hash}")

            cmd = ["git", "checkout", commit_hash]
            result = subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                if self.logger:
                    self.logger.warning(f"Change failed: {result.stderr.strip()}")
                return False

            if self.logger:
                self.logger.debug(f"Change succeeded: {commit_hash}")
            return True

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Change error: {str(e)}, traceback: {format_exc()}.")
            return False

    def process_dataset_item(self, item: dict) -> bool:
        """
        Process dataset item

        Args:
            item: dataset item

        Returns:
            bool: process succeeded
        """
        try:
            instance_id = item.get("instance_id")
            repo_name = item.get("repo")  # such as 'sqlfluff/sqlfluff'
            base_commit = item.get("base_commit")

            if not instance_id:
                if self.logger:
                    self.logger.error("Lack of instance_id")
                return False

            if not repo_name:
                if self.logger:
                    self.logger.error(f"Item {instance_id} lack of repo info")
                return False

            repo_url = self._convert_repo_to_url(repo_name)

            if self.logger:
                self.logger.debug(f"Process instance {instance_id}: {repo_name} -> {repo_url}")

            instance_dir = self.workspace_path / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)

            repo_dir = instance_dir / "repository"
            success = self.clone_repository(repo_url, repo_dir, base_commit)

            if success:
                info_file = instance_dir / "instance_info.txt"
                with open(info_file, "w", encoding="utf-8") as f:
                    f.write(f"Instance ID: {instance_id}\n")
                    f.write(f"Repository Name: {repo_name}\n")
                    f.write(f"Repository URL: {repo_url}\n")
                    f.write(f"Base Commit: {base_commit}\n")
                    f.write(f"Problem Statement: {item.get('problem_statement', '')}\n")
                    f.write(f"Created At: {item.get('created_at', '')}\n")

                if self.logger:
                    self.logger.info(f"Success process instance: {instance_id}")
                return True
            else:
                if self.logger:
                    self.logger.error(f"Failed to process instance: {instance_id}")
                return False

        except Exception as e:
            error_msg = f"Process data item error: {str(e)}, traceback: {format_exc()}."
            if self.logger:
                self.logger.error(error_msg)
            return False

    def load_and_process_all(self, max_items: Optional[int] = None) -> dict:
        """
        Load dataset items and process them.

        Args:
            max_items:max number of items to load(optional)

        Returns:
            dict: statics of results
        """
        if self.dataset is None:
            self.load_dataset()

        total_items = len(self.dataset)
        if max_items:
            total_items = min(total_items, max_items)

        successful_items = 0
        failed_items = 0

        if self.logger:
            self.logger.info(f"Start processing {total_items} items")

        for i, item in enumerate(self.dataset):
            if max_items and i >= max_items:
                break

            if self.logger:
                self.logger.debug(
                    f"Process {i+1}/{total_items}th item: {item.get('instance_id', 'unknown')}"
                )

            success = self.process_dataset_item(item)

            if success:
                successful_items += 1
            else:
                failed_items += 1

        result = {
            "total_processed": total_items,
            "successful": successful_items,
            "failed": failed_items,
            "success_rate": (
                (successful_items / total_items * 100) if total_items > 0 else 0
            ),
        }

        return result

    def get_instance_path(self, instance_id: str) -> Path:
        """Get path of instance"""
        return self.workspace_path / instance_id

    def get_repository_path(self, instance_id: str) -> Path:
        """Get path of repository for a specific instance"""
        return self.workspace_path / instance_id / "repository"

    def list_instances(self) -> list:
        """List all available instances in workspace"""
        instances = []
        if self.workspace_path.exists():
            for item in self.workspace_path.iterdir():
                if item.is_dir():
                    instances.append(item.name)
        return sorted(instances)

    def get_stats(self) -> dict:
        """Get statistics of a workspace"""
        instances = self.list_instances()
        cloned_repos = 0

        for instance_id in instances:
            repo_path = self.get_repository_path(instance_id)
            if repo_path.exists() and (repo_path / ".git").exists():
                cloned_repos += 1

        return {
            "total_instances": len(instances),
            "cloned_repositories": cloned_repos,
            "workspace_path": str(self.workspace_path.absolute()),
        }


class BuiltImageItem(NamedTuple):
    test_spec: TestSpec
    user_image_key: str = ''

class SWEBenchImageBuilder:
    __slots__ = [
        'client', 'dataset_name', 'split', 'namespace', 'logger',
        '_immutable_attrs',
        '_initialized',
        'full_dataset',
        'instance_to_image',
    ]
    DEFAULT_USER_PACKAGES = ['ripgrep', 'gettext']
    DEFAULT_ENV_IMAGE_TAG = 'latest'
    DEFAULT_TAG = 'latest'
    DEFAULT_MAX_RETRIES = 1
    DEFAULT_MAX_WORKERS = 2
    DEFAULT_NAMESPACE = 'swebench'

    def __init__(
        self,
        dataset_name: str = "SWE-bench/SWE-bench_Lite",
        split: str = "test",
        namespace: str = DEFAULT_NAMESPACE,
        logger: Optional[Union[logging.Logger, CustomLogger]] = None,
    ):
        """
        Initialize the image builder and build all required images.

        Args:
            dataset_name: Name of the dataset to use
            split: Split to use (dev/test)
            namespace: Namespace for images, default is 'swebench'. Please use the same namespace as the swebench.harness.run_evaluation.
            logger: Logger instance (can be either logging.Logger or CustomLogger from logger.py)
        """
        self._immutable_attrs = {
            'client', 'dataset_name', 'split', 'namespace', 'logger', 'full_dataset'
        }

        self._initialized = False
        
        self.client = docker.from_env()
        self.dataset_name = dataset_name
        self.split = split
        self.namespace = namespace
        self.logger = logger
        if self.logger is None:
            log_path = Path(__file__).resolve().parent.parent.parent.parent / 'logs' / 'build_images'
            self.logger = logging.getLogger("SWEBenchImageBuilder")
            self.logger.setLevel(logging.INFO)
            logfile = log_path / f"build_images_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.log"
            _handler = logging.FileHandler(logfile)
            _handler.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s[%(module)s:%(funcName)s] %(message)s"))
            self.logger.addHandler(_handler)
            setattr(self.logger, "log_file", logfile.absolute().__str__())
        elif not hasattr(self.logger, "log_file"):
            setattr(self.logger, "log_file", 'unkown')
            self.logger.warning("logger has no log_file attribute, which may cause some problems.")

        self.full_dataset = load_swebench_dataset(self.dataset_name, self.split)
        self.instance_to_image = {}
        self._initialized = True
    
    def __setattr__(self, name, value):
        """
        override __setattr__ to prevent immutable attributes to be modified
        """
        if hasattr(self, '_initialized') and \
                self._initialized and \
                name in self._immutable_attrs:
            raise AttributeError(f"Cannot modify immutable attribute '{name}'")
        super().__setattr__(name, value)

    def load_images(self, instance_ids: List[str]=None):
        self.instance_to_image = self.load_instance_to_image(instance_ids=instance_ids)

    def build_target_images(self, 
        instance_ids: List[str]=None, 
        force_rebuild: bool=False,
        tag: str=DEFAULT_TAG, 
        env_image_tag: str=DEFAULT_ENV_IMAGE_TAG,
        user_packages: List[str]=DEFAULT_USER_PACKAGES,
        max_retries: int=DEFAULT_MAX_RETRIES,
        max_workers: int=DEFAULT_MAX_WORKERS):
        """
        Build target images
        Args:
            instance_ids: Instance ID list of corresponding dataset. Build full instance images when list is [] or None
            force_rebuild: Whether to force rebuild
            tag: User defined image tag
            env_image_tag: Environment image tag
            user_packages: Packages of linux cli tools in user defined images
            max_retries: Max retries
            max_workers: Max workers
        """
        dataset_to_build = self._filter_dataset_to_build(
            self.full_dataset,
            instance_ids,
            self.client,
            force_rebuild,
            self.namespace,
            tag,
            env_image_tag,
        )

        self.instance_to_image = self.load_instance_to_image(instance_ids)

        if self.instance_to_image:
            self.logger.info(
                f"Filtered out {len(self.instance_to_image)} existing instances from local docker."
            )
            existing_instance_ids = set(self.instance_to_image.keys())
            dataset_to_build = [
                inst
                for inst in dataset_to_build
                if inst["instance_id"] not in existing_instance_ids
            ]
            self.logger.info(f"Remaining instances to process: {len(dataset_to_build)}")

        if len(dataset_to_build) == 0:
            self.logger.info("All images exist. Nothing left to build.")
            return

        self.logger.info(f"{'='*50}\nStart building images for {len(dataset_to_build)} instances\n{'='*50}")

        _successful, _failed = self._build_instance_images_with_retry(dataset_to_build, force_rebuild, tag, env_image_tag, max_retries, max_workers)
        self.logger.info(f"Build images for {len(_successful)} instances successfully, {len(_failed)} instances failed\n{'='*50}")
        if len(_failed) > 0:
            self.logger.warning(f"failed instances: {[x[0].instance_id for x in _failed]}")

        if len(user_packages) == 0:
            return
        
        if _successful:
            self.logger.info(f"Start building {len(_successful)} user images...")
            self._build_many_user_images(_successful, user_packages, tag, max_workers)

    #  _get_instance_to_image_path ：Save the name of packaged images
    def _get_instance_to_image_path(self) -> Path:
        here = Path(__file__).resolve().parent
        target_path = here / "instance_to_image.json"
        return target_path

    def load_instance_to_image(self, instance_ids: List[str]=None):
        """Load the mapping of instances to images"""
        target_path = self._get_instance_to_image_path()
        self.logger.debug(f"In load_instance_to_image, target_path: {target_path}.")
        existing_instance_to_image_dict = self._get_all_existing_instance_to_image(
            self.full_dataset,
            self.client,
            self.namespace
        )
        self.logger.debug(f"In load_instance_to_image, existing_instance_to_image_dict: {existing_instance_to_image_dict}.")
        di = {}
        
        self.logger.info(f"start updating instance_to_image.json")
        try: 
            # Load full
            with target_path.open("r+", encoding="utf-8") as f:
                di = json.load(f)
                ids_to_remove = set()
                self.logger.debug(f"In load_instance_to_image, loaded di: {di}.")

                # Remove those in json but not exists
                for instance_id in di.keys():               
                    if instance_id not in existing_instance_to_image_dict:
                        ids_to_remove.add(instance_id)
                        
                self.logger.debug(f"In load_instance_to_image, instance_ids for images to remove: {ids_to_remove}")
                
                for instance_id in ids_to_remove:              
                    try:
                        self.logger.debug(f"In load_instance_to_image, remove user image: {di[instance_id]['user_image_key']}")
                        self.client.images.remove(di[instance_id]['user_image_key'], force=True)
                    except docker.errors.ImageNotFound:
                        pass
                    del di[instance_id]

                di = {**di, **existing_instance_to_image_dict}
                self.logger.debug(f"In load_instance_to_image, after merge di: {di}.")
                f.seek(0)
                json.dump(di, f, ensure_ascii=False, indent=4)
                f.truncate()  
        except FileNotFoundError as e:
            self.logger.warning(f"In load_instance_to_image, failed to load instance_to_image.json: {e}")
            with target_path.open("w", encoding="utf-8") as f:
                json.dump(existing_instance_to_image_dict, f, ensure_ascii=False, indent=4)
            return existing_instance_to_image_dict
        
        if instance_ids:
            return {k: v for k, v in di.items() if k in instance_ids}
        self.logger.debug(f"In load_instance_to_image, instance_to_image: {di}.")
        self.instance_to_image = di
        return di
    
    def _build_instance_images_with_retry(self, dataset_to_build: List, 
        force_rebuild: bool=False, 
        tag: str=DEFAULT_TAG, 
        env_image_tag: str=DEFAULT_ENV_IMAGE_TAG,
        max_retries: int=DEFAULT_MAX_RETRIES,
        max_workers: int=DEFAULT_MAX_WORKERS):
        """
        Build target images with retry strategy.
        
        Args:
            force_rebuild: Whether to force rebuild
            tag: Image tag
            env_image_tag: Environment image tag
            max_retries: Max retries
            max_workers: Max workers
            
        Returns:
            tuple: (Success, Failed)
        """
        build_params = {
            'client': self.client,
            'dataset': dataset_to_build,
            'force_rebuild': force_rebuild,
            'max_workers': max_workers,
            'namespace': self.namespace,
            'tag': tag,
            'env_image_tag': env_image_tag,
        }

        with redirect_swebench_prints(self.logger):
            _successful, _failed = patched_build_instance_images(
                **build_params,
                custom_logger=self.logger
            )
        for attempt in range(max_retries):
            if len(_failed) == 0:
                break

            self.logger.info(f"Retry {attempt + 1}/{max_retries}: building images for {len(_failed)} instances...")
            # _failed: tuple list，each item: (TestSpec, docker.Client, None, False)
            _retry_instances = [x[0].instance_id for x in _failed]
            _retry_dataset = [x for x in dataset_to_build if x['instance_id'] in _retry_instances]
            build_params['dataset'] = _retry_dataset
            with redirect_swebench_prints(self.logger):
                retry_successful, retry_failed = patched_build_instance_images(
                    **build_params,
                    custom_logger=self.logger
                )
            
            _successful.extend(retry_successful)
            _failed = retry_failed
        
        return _successful, _failed
    
    def _update_instance_to_image(self, test_specs: List[BuiltImageItem]):
        di = self.load_instance_to_image()
        for spec in test_specs:         
            if spec.test_spec.instance_id not in di:
                di[spec.test_spec.instance_id] = {
                    "base_image_key": spec.test_spec.base_image_key,
                    "env_image_key": spec.test_spec.env_image_key,
                    "instance_image_key": spec.test_spec.instance_image_key,
                }
                di[spec.test_spec.instance_id]['user_image_key'] = spec.user_image_key or spec.test_spec.instance_image_key
                continue
            
            di[spec.test_spec.instance_id]['base_image_key'] = spec.test_spec.base_image_key
            di[spec.test_spec.instance_id]['env_image_key'] = spec.test_spec.env_image_key
            di[spec.test_spec.instance_id]['instance_image_key'] = spec.test_spec.instance_image_key
            di[spec.test_spec.instance_id]['user_image_key'] = spec.user_image_key or spec.test_spec.instance_image_key

        self.instance_to_image = di
        target_path = self._get_instance_to_image_path()
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(di, f, ensure_ascii=False, indent=4)

    # _build_user_image：Add user defined tools to swebench standard instance_image
    def _build_user_image(self, spec: TestSpec, client: docker.DockerClient, packages: List[str], tag: str='latest'):
        _packages = " ".join(packages)
        dockerfile = _DOCKERFILE_USER_IMAGE_PY.format(image_key=spec.instance_image_key, 
        apt_packages=_packages)
        _dockerfile_obj = io.BytesIO(dockerfile.encode())
        try:
            _instance_image_name = spec.instance_image_key.split(":")[0]
            _image_key = f"{_instance_image_name}.user:{tag}"
            _image, _logs = client.images.build(fileobj=_dockerfile_obj, tag=_image_key)
            building_warnings = []
            for log_entry in _logs:
                    if 'stream' in log_entry:
                        log_line = log_entry['stream'].strip()
                        warning_patterns = [
                            'warning:', 'warn:', 'w:',
                            'failed to fetch', 
                            'unable to locate package',
                            'has no installation candidate',
                            'ignored',
                            'error'
                        ]
                        if any(pattern in log_line.lower() for pattern in warning_patterns):
                            building_warnings.append(log_line)
            if building_warnings:
                warning_msg = '\n'.join(building_warnings)
                self.logger.error(f"docker build warning: {warning_msg}")
                try:
                    client.images.remove(_image_key, force=True)
                except Exception as e:
                    self.logger.warning(f"Error removing image {_image_key}: {e}\nTraceback: {format_exc()}\n")
                return None
            self.logger.info(f"✓ Successfully built user image for {spec.instance_id}: {_image.tags[0]}")
            return BuiltImageItem(spec, _image_key)
        except Exception as e:
            self.logger.error(f"Error building user image for {spec.instance_id}: {e}\nTraceback: {format_exc()}\n")
            return None

    def _build_many_user_images(self, successful_pairs, user_packages: List[str], 
    tag: str=DEFAULT_TAG, max_workers: int=DEFAULT_MAX_WORKERS):
        """
        Build user images
        Params：
            successful_pairs: Successful pairs, each item: (TestSpec, docker.Client, None, False)
            user_packages: Package of cli tools in user defined image
            max_workers: Max workers
        """
        if user_packages is None or len(user_packages) == 0:
            return

        results: List[Union[BuiltImageItem, None]] = []
        max_workers = max(1, int(max_workers))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._build_user_image, s[0], s[1], user_packages, tag) for s in successful_pairs]
            for f in as_completed(futures):
                item = f.result()
                if item:
                    results.append(item)
        results = [item for item in results if item is not None]
        if results:
            self._update_instance_to_image(results)

    def _filter_dataset_to_build(
        self,
        dataset: list,
        instance_ids: list | None,
        client: docker.DockerClient,
        force_rebuild: bool = False,
        namespace: str = DEFAULT_NAMESPACE,
        tag: str = DEFAULT_TAG,
        env_image_tag: str = DEFAULT_ENV_IMAGE_TAG,
    ):
        """
        Filter the dataset to only include instances that need to be built.

        Args:
            dataset (list): List of instances (usually all of SWE-bench dev/test split)
            instance_ids (list): List of instance IDs to build.
            client (docker.DockerClient): Docker client.
            force_rebuild (bool): Whether to force rebuild all images.
        Returns:
            list: Instance list to build.
        """
        existing_images = {tag for i in client.images.list(all=False) for tag in i.tags}
        data_to_build = []

        if instance_ids is None:
            instance_ids = [instance['instance_id'] for instance in dataset]

        not_in_dataset = set(instance_ids).difference(
            set([instance['instance_id'] for instance in dataset])
        )
        if not_in_dataset:
            raise ValueError(f"Instance IDs not found in dataset: {not_in_dataset}")

        for instance in dataset:
            if instance['instance_id'] not in instance_ids:
                continue

            # Check if the instance needs to be built (based on force_rebuild flag and existing images)
            spec = make_test_spec(
                instance,
                namespace=namespace,
                instance_image_tag=tag,
                env_image_tag=env_image_tag,
            )
            if force_rebuild:
                data_to_build.append(instance)
            elif (spec.instance_image_key not in existing_images) or (not self._is_user_image_exist(existing_images, spec)):
                data_to_build.append(instance)

        return data_to_build

    def _get_all_existing_instance_to_image(self,
        dataset: list,
        client: docker.DockerClient,
        namespace: str = DEFAULT_NAMESPACE,
        tag: str = DEFAULT_TAG,
        env_image_tag: str = DEFAULT_ENV_IMAGE_TAG):
        self.logger.info(f"Getting all existing images...")
        existing_images = {tag for i in client.images.list(all=False) for tag in i.tags}
        self.logger.info(f"start update existing_instance_to_image_dict...")
        existing_instance_to_image_dict = {}

        for instance in dataset:
            spec = make_test_spec(
                instance,
                namespace=namespace,
                instance_image_tag=tag,
                env_image_tag=env_image_tag,
            )
            if spec.instance_image_key in existing_images and self._is_user_image_exist(existing_images, spec):
                existing_instance_to_image_dict[instance['instance_id']] = {
                    "base_image_key": spec.base_image_key,
                    "env_image_key": spec.env_image_key,
                    "instance_image_key": spec.instance_image_key,
                    "user_image_key": spec.instance_image_key.split(':')[0] + ".user:" + spec.instance_image_tag,
                }
        self.logger.info(f"end with updating existing_instance_to_image_dict.")
        return existing_instance_to_image_dict

    def _is_user_image_exist(self, existing_images, instance: TestSpec) -> bool:
        instance_image_key = instance.instance_image_key
        user_image_key = f"{instance_image_key.split(':')[0]}.user:{instance.instance_image_tag}"
        return (user_image_key in existing_images)

    def get_image_info(self, instance_id: str) -> str:
        """
        Get the Docker image name for a given instance_id.

        Args:
            instance_id: The instance ID to look up

        Returns:
            The Docker image information Dict(str, str) for the instance.
            Keys: `base_image_key`, `env_image_key`, `instance_image_key`, `user_image_key`.

        Raises:
            KeyError: If instance_id is not found in the dataset
        """
        if instance_id not in self.instance_to_image.keys():
            raise KeyError(f"Instance ID '{instance_id}' not found in dataset")

        return self.instance_to_image[instance_id]

    def get_image_name(self, instance_id: str) -> str:
        """
        Get the user image key for a given instance_id.
        """
        if instance_id not in self.instance_to_image.keys() or 'user_image_key' not in self.instance_to_image[instance_id]:
            raise KeyError(f"Instance ID '{instance_id}' user_image_key not found in dataset")
        return self.instance_to_image[instance_id]['user_image_key']
        
    # only for test
    def get_build_status(self, instance_id: str) -> str:
        """
        Get the build status for a given instance_id.

        Args:
            instance_id: The instance ID to check

        Returns:
            'successful', 'failed'
        """
        if instance_id not in self.instance_to_image.keys():
            return "failed"
        return "successful"

    def check_user_package_exist(self, packages: List[str]=DEFAULT_USER_PACKAGES) -> List[str]:
        """Check whether the user package bendi exists."""

        invalid_instance_ids = []
        if self.instance_to_image is None or len(self.instance_to_image) == 0:
            self.instance_to_image = self.load_instance_to_image()
        
        for instance_id, instance_info in self.instance_to_image.items():
            user_image_key = instance_info['user_image_key']
            self.logger.info(f"checking user packages for: {instance_id}")
            try:
                container = self.client.containers.run(
                    user_image_key,
                    command="sleep infinity",
                    detach=True,
                    working_dir="/workspace",
                )
            except Exception as e:
                invalid_intance_ids.append(instance_id)
                continue

            pattern = '|'.join(packages)
            _cmd = f'apt list --installed | grep -E \'({pattern})\''
            _shell_cmd = f'bash -c "{_cmd}"'
            _, output = container.exec_run(
                cmd=_shell_cmd,
                workdir="/workspace",
            )

            pattern_group = ''
            patterns = []
            for p in packages:
                patterns.append(f'^{p}.*\\[installed\\]$')
            pattern_group = '|'.join(patterns)
            regex = re.compile(pattern_group, flags=re.MULTILINE)
            if isinstance(output, bytes):
                output_str = output.decode('utf-8')
            else:
                output_str = str(output)
            matches = regex.findall(output_str)
            if len(matches) != len(packages):
                self.logger.warning(f"user packages not installed for {instance_id}.\ninstalled packages: {matches}\nexpected packages: {packages}")
                invalid_instance_ids.append(instance_id)
                container.stop()
                try:
                    self.client.images.remove(user_image_key, force=True)
                except Exception as e:
                    self.logger.warning(f"Error removing image {user_image_key}: {e}\nTraceback: {format_exc()}\n")
            container.stop()
            container.remove()
        self.logger.info(f"nonexistent instance ids: {invalid_instance_ids}")
        self.logger.info(f'update instance_to_image again...')
        self.instance_to_image = self.load_instance_to_image()
        return invalid_instance_ids


    
    
