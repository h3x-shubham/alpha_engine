"""
YAML configuration loader with validation and environment variable expansion.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("alpha_engine.config")


class ConfigLoader:
    """
    Loads and merges YAML configuration files.

    Supports:
    - Environment variable substitution via ${VAR_NAME} syntax
    - Merging multiple YAML files (later files override earlier ones)
    - Dot-notation access for nested keys
    """

    def __init__(self, config_dir: str | Path = "config") -> None:
        self.config_dir = Path(config_dir)
        self._data: dict[str, Any] = {}

    def load(self, *filenames: str) -> "ConfigLoader":
        """Load one or more YAML files from the config directory."""
        for fname in filenames:
            path = self.config_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, "r") as f:
                raw = f.read()
            # Expand environment variables: ${VAR_NAME}
            raw = self._expand_env_vars(raw)
            parsed = yaml.safe_load(raw) or {}
            self._data = self._deep_merge(self._data, parsed)
            logger.info("Loaded config: %s", path)
        return self

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Access nested config values via dot notation.

        Example:
            config.get("model.hyperparameters.learning_rate")
        """
        keys = dotted_key.split(".")
        node = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def require(self, dotted_key: str) -> Any:
        """Access a config value, raising if it does not exist."""
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"Required config key missing: {dotted_key}")
        return value

    @property
    def data(self) -> dict[str, Any]:
        """Return the full config dictionary."""
        return self._data

    # ── Private helpers ───────────────────────

    @staticmethod
    def _expand_env_vars(text: str) -> str:
        """Replace ${VAR_NAME} with environment variable values."""
        pattern = re.compile(r"\$\{(\w+)\}")

        def _replacer(match: re.Match) -> str:
            var = match.group(1)
            val = os.environ.get(var)
            if val is None:
                logger.warning("Environment variable not set: %s", var)
                return match.group(0)
            return val

        return pattern.sub(_replacer, text)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override dict into base dict."""
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def __repr__(self) -> str:
        return f"ConfigLoader(config_dir={self.config_dir}, keys={list(self._data.keys())})"
