import importlib
import logging
import os

from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger("task loader")


def import_all_tasks(root_dir, base_module="tasks") -> None:
    for root, dirs, files in os.walk(root_dir):
        rel_path = os.path.relpath(root, start=root_dir).replace(os.sep, ".")
        current_module_base = (
            f"{base_module}.{rel_path}" if rel_path != "." else base_module
        )

        for file in files:
            if file == "task.py":
                full_module_path = f"{current_module_base}.task"
                try:
                    importlib.import_module(full_module_path)
                    logger.info(f"✅   Imported task module: {full_module_path}")
                except Exception as e:
                    logger.error(f"❌   Failed to import {full_module_path}: {e}")
