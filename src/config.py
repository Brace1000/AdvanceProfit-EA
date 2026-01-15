"""
Configuration management for AdvanceProfit-EA.

This module handles loading and accessing configuration from YAML files.
Priority: config.local.yaml > config.yaml
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager that loads settings from YAML files."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_path: Optional path to config file. If None, will look for
                        config.local.yaml first, then config.yaml in project root.
        """
        self.project_root = Path(__file__).parent.parent

        if config_path:
            self.config_path = config_path
        else:
            # Try local config first, fall back to default
            local_config = self.project_root / "config.local.yaml"
            default_config = self.project_root / "config.yaml"

            if local_config.exists():
                self.config_path = local_config
            elif default_config.exists():
                self.config_path = default_config
            else:
                raise FileNotFoundError(
                    f"No config file found. Please create {default_config} "
                    f"or {local_config}"
                )

        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., "model.path" or "trading.risk.risk_percent")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = Config()
            >>> config.get("model.path")
            'xgb_eurusd_h1.pkl'
            >>> config.get("trading.risk.risk_percent")
            1.0
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., "model", "trading", "api")

        Returns:
            Configuration section as dictionary
        """
        return self._config.get(section, {})

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get_section("model")

    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get_section("training")

    @property
    def hpo(self) -> Dict[str, Any]:
        """Get HPO configuration."""
        return self.get_section("hpo")

    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get_section("api")

    @property
    def trading(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.get_section("trading")

    @property
    def backtesting(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.get_section("backtesting")

    @property
    def mlflow(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.get_section("mlflow")

    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.get_section("paths")

    def get_model_path(self) -> Path:
        """Get full path to model file."""
        model_path = self.get("model.path", "xgb_eurusd_h1.pkl")
        return self.project_root / model_path

    def get_data_path(self, data_type: str = "raw") -> Path:
        """
        Get path to data directory.

        Args:
            data_type: Type of data ("raw" or "processed")

        Returns:
            Path to data directory
        """
        if data_type == "raw":
            path = self.get("paths.data_raw", "data/raw")
        elif data_type == "processed":
            path = self.get("paths.data_processed", "data/processed")
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        return self.project_root / path

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config(path={self.config_path})"


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None, reload: bool = False) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Optional path to config file
        reload: If True, reload config even if already loaded

    Returns:
        Config instance
    """
    global _config

    if _config is None or reload:
        _config = Config(config_path)

    return _config


# Convenience functions
def get(key_path: str, default: Any = None) -> Any:
    """Get config value using dot notation."""
    return get_config().get(key_path, default)


def get_section(section: str) -> Dict[str, Any]:
    """Get entire config section."""
    return get_config().get_section(section)


if __name__ == "__main__":
    # Example usage
    config = Config()
    print(f"Model path: {config.get('model.path')}")
    print(f"Risk percent: {config.get('trading.risk.risk_percent')}")
    print(f"ML enabled: {config.get('trading.ml.use_predictions')}")
    print(f"Full model config: {config.model}")
