from src.managers.image_builder.build_image import SWEBenchImageBuilder
import yaml
from pathlib import Path
from src.managers.log.logger import ImageBuilderLogger
from typing import List


def load_instance_ids(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]



class BuildImage:
    def __init__(self, instance_ids: List[str] = None):
        self.instance_ids = instance_ids or []
        self.config = self._load_config()
        self.logger = ImageBuilderLogger(
            log_base_path=self.config.get("log", {}).get("image_builder", "workspace/image_logs"),
            console_output=True,
        )
        self.builder = self._init_builder()


    def _init_builder(self):
        dataset_name = self.config.get("dataset", {}).get("name", "princeton-nlp/SWE-bench_Lite")
        dataset_split = self.config.get("dataset", {}).get("split", "dev")
        return SWEBenchImageBuilder(
            dataset_name=dataset_name,
            split=dataset_split,
            logger=self.logger,
        )
    
    def _load_config(self):
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return  yaml.safe_load(f) or {}

    def _build(self):
        max_workers = self.config.get("builder", {}).get("max_workers", 3)
        max_retries = self.config.get("builder", {}).get("max_retries", 1)
        self.builder.build_target_images(instance_ids=self.instance_ids, max_workers=max_workers, max_retries=max_retries)

if __name__ == "__main__":
    #instance_ids = ["django__django-10097", "django__django-10880"]
    file_path = "" # fill in the list path
    instance_ids = load_instance_ids(file_path=file_path)
    print(instance_ids)
    build_image = BuildImage(instance_ids)
    build_image._build()