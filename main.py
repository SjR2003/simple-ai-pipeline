import os
import mlflow
import yaml
import argparse
from pathlib import Path
import warnings

from utils.logger import setup_logger
from core.task_registry import get_task
from core.base_task import BaseTask
from core.task_loader import import_all_tasks


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def init_mlflow(exp):
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message="Filesystem tracking backend.*"
    )

    out = Path(__file__).parent
    mlflow.set_tracking_uri(f"file:{out / 'mlruns'}")
    mlflow.set_experiment(exp)
    mlflow.start_run(run_name="pipeline_run")


def run_task_from_config(config_path, result=None):
    if not os.path.isfile(config_path):
        logger.error(f"⚠️   Config file not found: {config_path}")
        return None

    config = load_yaml_config(config_path)
    mlflow.log_artifact(config_path, f"config file - {os.path.basename(config_path)}")

    task_name = config.get("task_name")
    if not task_name:
        logger.error(f"⚠️  {config_path} — missing 'task_name' field.")
        return None

    logger.info(f"🔍   Processing task config: {config_path} ({task_name})")

    TaskClass = get_task(task_name)
    if not TaskClass:
        logger.error(f"❌   Task class not found: {task_name}")
        return None

    if not issubclass(TaskClass, BaseTask):
        logger.error(f"⚠️   Task {task_name} is not a subclass of BaseTask.")
        return None

    task = TaskClass(config, result)
    task.run()
    logger.info(f"✅   Task finished: {task_name}\n")

    return task.result


def run_all_tasks(config_dir: str):
    task_order_path = os.path.join(config_dir, "master_config.yaml")
    if not os.path.isfile(task_order_path):
        logger.error(f"❌   master_config.yaml not found in '{config_dir}'")
        return

    with open(task_order_path, "r") as f:
        config = yaml.safe_load(f)
        task_order = config.get("task_order", [])
        experiment = config.get("experiment", "ai pipeline")

    config_dir = os.path.join(config_dir, "tasks")
    import_all_tasks(root_dir="tasks", base_module="tasks")

    if not task_order:
        logger.warning("⚠️   No tasks specified in master_config.yaml")
        return

    result = None
    init_mlflow(experiment)
    for task_file in task_order:
        result = run_task_from_config(
            os.path.join(config_dir, task_file), result=result
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YAML-defined pipeline tasks.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default="configs",
        help="Path to the directory containing task YAML configs.",
    )

    args = parser.parse_args()
    logger = setup_logger(name="main")
    logger.info("🏭   Pipeline started!")

    run_all_tasks(args.config)
