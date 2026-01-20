"""
Configuration management system for Code Autocomplete Assistant.

Supports global and project-specific configurations with hot-reload capability.
Addresses Requirements 7.1, 7.2, 7.3, 7.5.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class ModelProfile(str, Enum):
    """Available model profiles for different training stages."""
    BASE = "base"
    STAGE_A = "stage_a" 
    STAGE_B = "stage_b"


class ModelConfig(BaseModel):
    """Model-specific configuration."""
    profile: ModelProfile = ModelProfile.BASE
    max_tokens: int = Field(default=50, ge=1, le=500)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    n_suggestions: int = Field(default=3, ge=1, le=10)
    timeout_ms: int = Field(default=2000, ge=100, le=10000)


class AutocompleteConfig(BaseModel):
    """Autocomplete engine configuration."""
    enabled: bool = True
    aggressiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    min_prefix_length: int = Field(default=2, ge=0, le=20)
    max_suffix_length: int = Field(default=100, ge=0, le=1000)


class CommentGeneratorConfig(BaseModel):
    """Comment-to-code generator configuration."""
    enabled: bool = True
    max_output_lines: int = Field(default=50, ge=1, le=200)
    include_imports: bool = True
    validate_syntax: bool = True


class CDKConfig(BaseModel):
    """AWS CDK-specific configuration."""
    prioritize_cdk_patterns: bool = True
    include_best_practices: bool = True
    default_encryption: bool = True
    custom_patterns: list[str] = Field(default_factory=list)


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=8)
    log_level: str = "INFO"


class AppConfig(BaseModel):
    """Main application configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    autocomplete: AutocompleteConfig = Field(default_factory=AutocompleteConfig)
    comment_generator: CommentGeneratorConfig = Field(default_factory=CommentGeneratorConfig)
    cdk: CDKConfig = Field(default_factory=CDKConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    # Paths
    models_dir: Path = Field(default=Path("models"))
    datasets_dir: Path = Field(default=Path("datasets"))
    logs_dir: Path = Field(default=Path("logs"))


class ConfigManager:
    """
    Configuration manager with support for global and project-specific configs.
    
    Configuration precedence (highest to lowest):
    1. Project-specific config (.code-assistant.yaml in project root)
    2. Global config (~/.code-assistant/config.yaml)
    3. Default values
    """
    
    def __init__(self):
        self._config: AppConfig = AppConfig()
        self._global_config_path = Path.home() / ".code-assistant" / "config.yaml"
        self._project_config_cache: Dict[str, AppConfig] = {}
        
    def get_config(self, project_path: Optional[Union[str, Path]] = None) -> AppConfig:
        """
        Get configuration for a specific project or global config.
        
        Args:
            project_path: Path to project directory. If None, returns global config.
            
        Returns:
            Merged configuration with project-specific overrides applied.
        """
        if project_path is None:
            return self._get_global_config()
            
        project_path = Path(project_path).resolve()
        project_key = str(project_path)
        
        # Check cache first
        if project_key in self._project_config_cache:
            return self._project_config_cache[project_key]
            
        # Load and merge configurations
        config = self._load_project_config(project_path)
        self._project_config_cache[project_key] = config
        
        return config
    
    def reload_config(self, project_path: Optional[Union[str, Path]] = None) -> AppConfig:
        """
        Reload configuration from disk, clearing cache.
        
        Args:
            project_path: Path to project directory. If None, reloads global config.
            
        Returns:
            Reloaded configuration.
        """
        if project_path is None:
            self._config = self._load_global_config()
            return self._config
            
        project_path = Path(project_path).resolve()
        project_key = str(project_path)
        
        # Clear cache and reload
        if project_key in self._project_config_cache:
            del self._project_config_cache[project_key]
            
        return self.get_config(project_path)
    
    def _get_global_config(self) -> AppConfig:
        """Get cached global configuration."""
        return self._config
    
    def _load_global_config(self) -> AppConfig:
        """Load global configuration from disk."""
        if not self._global_config_path.exists():
            return AppConfig()
            
        try:
            with open(self._global_config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            return AppConfig(**data)
        except Exception as e:
            print(f"Warning: Failed to load global config: {e}")
            return AppConfig()
    
    def _load_project_config(self, project_path: Path) -> AppConfig:
        """Load project-specific configuration with global config as base."""
        # Start with global config
        global_config = self._get_global_config()
        
        # Look for project config
        project_config_path = project_path / ".code-assistant.yaml"
        if not project_config_path.exists():
            return global_config
            
        try:
            with open(project_config_path, 'r') as f:
                project_data = yaml.safe_load(f) or {}
                
            # Merge with global config
            merged_data = self._merge_configs(
                global_config.model_dump(), 
                project_data
            )
            
            return AppConfig(**merged_data)
        except Exception as e:
            print(f"Warning: Failed to load project config: {e}")
            return global_config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def save_global_config(self, config: AppConfig) -> None:
        """Save global configuration to disk."""
        self._global_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self._global_config_path, 'w') as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)
            
        self._config = config
    
    def create_project_config_template(self, project_path: Union[str, Path]) -> Path:
        """Create a template project configuration file."""
        project_path = Path(project_path)
        config_path = project_path / ".code-assistant.yaml"
        
        template_config = {
            "model": {
                "profile": "base",
                "temperature": 0.2,
                "n_suggestions": 3
            },
            "autocomplete": {
                "enabled": True,
                "aggressiveness": 0.5
            },
            "comment_generator": {
                "enabled": True,
                "validate_syntax": True
            },
            "cdk": {
                "prioritize_cdk_patterns": True,
                "custom_patterns": []
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False)
            
        return config_path


# Global configuration manager instance
config_manager = ConfigManager()


def get_config(project_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Convenience function to get configuration."""
    return config_manager.get_config(project_path)


def reload_config(project_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Convenience function to reload configuration."""
    return config_manager.reload_config(project_path)