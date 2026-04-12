import os
from typing import Optional

class PathManager:
    """
    Centralized path management for the project.
    Resolves relative paths to absolute paths based on the project root.
    """
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def get_data_path(cls, sub_path: str = "") -> str:
        return os.path.join(cls.ROOT_DIR, "data", sub_path)

    @classmethod
    def get_config_path(cls, filename: Optional[str] = None) -> str:
        base = os.path.join(cls.ROOT_DIR, "configs")
        return os.path.join(base, filename) if filename else base

    @classmethod
    def get_log_path(cls, filename: Optional[str] = None) -> str:
        base = os.path.join(cls.ROOT_DIR, "logs")
        return os.path.join(base, filename) if filename else base

    @classmethod
    def get_model_save_path(cls, filename: str) -> str:
        return os.path.join(cls.ROOT_DIR, "models", "saved", filename)

# Ensure saved models directory exists
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "saved"), exist_ok=True)
