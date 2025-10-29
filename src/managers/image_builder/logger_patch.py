"""
User defined logger for swebench
"""
import logging
from typing import Union, Optional

from swebench.harness.docker_utils import remove_image
from src.managers.log.logger import Logger as CustomLogger

# 导入原始的 swebench 函数
from swebench.harness.docker_build import build_instance_image, run_threadpool


def patched_build_instance_images(
    client,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    namespace: str = None,
    tag: str = None,
    env_image_tag: str = None,
    custom_logger: Optional[Union[logging.Logger, CustomLogger]] = None,  # 新增参数
):
    """
    Monkey patched version of build_instance_images that supports custom logger
    
    Args:
        client: Docker client
        dataset: List of test specs or dataset to build images for
        force_rebuild: Whether to force rebuild the images even if they already exist
        max_workers: Maximum number of worker threads
        namespace: Namespace for images
        tag: Tag for images
        env_image_tag: Environment image tag
        custom_logger: Custom logger to use (instead of creating new ones)
    """
    from swebench.harness.docker_build import make_test_spec, build_env_images

    test_specs = []
    for instance in dataset:
        spec = make_test_spec(
            instance,
            namespace=namespace,
            instance_image_tag=tag,
            env_image_tag=env_image_tag,
        )
        test_specs.append(spec)

    if force_rebuild:
        for spec in test_specs:
            remove_image(client, spec.instance_image_key, "quiet")

    _, env_failed = build_env_images(client, test_specs, force_rebuild, max_workers)

    if len(env_failed) > 0:
        # Don't build images for instances that depend on failed-to-build env images
        dont_run_specs = [
            spec for spec in test_specs if spec.env_image_key in env_failed
        ]
        test_specs = [
            spec for spec in test_specs if spec.env_image_key not in env_failed
        ]
        if custom_logger:
            custom_logger.info(
                f"Skipping {len(dont_run_specs)} instances - due to failed env image builds"
            ) 
        else:
            print(f"Skipping {len(dont_run_specs)} instances - due to failed env image builds")

    if custom_logger:
        custom_logger.info(f"Building instance images for {len(test_specs)} instances")
    else:
        print(f"Building instance images for {len(test_specs)} instances")
    
    successful, failed = [], []

    if custom_logger:
        payloads = [(spec, client, custom_logger, False) for spec in test_specs]
    else:
        payloads = [(spec, client, None, False) for spec in test_specs]

    successful, failed = run_threadpool(build_instance_image, payloads, max_workers)

    if len(failed) == 0:
        if custom_logger:
            custom_logger.info("All instance images built successfully.")
        else:
            print("All instance images built successfully.")
    else:
        if custom_logger:
            custom_logger.warning(f"{len(failed)} instance images failed to build.")
        else:
            print(f"{len(failed)} instance images failed to build.")
    return successful, failed


def apply_logger_patch():
    """应用 logger patch"""
    import swebench.harness.docker_build as docker_build_module

    original_build_instance_images = docker_build_module.build_instance_images

    docker_build_module.build_instance_images = patched_build_instance_images
    
    return original_build_instance_images


def restore_logger_patch(original_function):
    """Recover original logger actions"""
    import swebench.harness.docker_build as docker_build_module
    docker_build_module.build_instance_images = original_function


# 上下文管理器版本
class LoggerPatch:
    """Context manager"""
    
    def __init__(self, logger: Optional[Union[logging.Logger, CustomLogger]] = None):
        self.logger = logger
        self.original_function = None
        
    def __enter__(self):
        self.original_function = apply_logger_patch()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_function:
            restore_logger_patch(self.original_function)
