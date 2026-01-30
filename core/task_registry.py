# core/task_registry.py
TASKS = {}


def register_task(name):
    def wrapper(cls):
        if name in TASKS:
            raise ValueError(f"Task '{name}' is already registered.")
        TASKS[name] = cls
        return cls

    return wrapper


def get_task(name):
    return TASKS.get(name)
