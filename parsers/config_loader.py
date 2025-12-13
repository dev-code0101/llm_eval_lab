"""
Configuration loader for YAML config files.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics"""
    response_relevance: bool = True
    response_completeness: bool = True
    hallucination: bool = True
    rouge: bool = True
    
    # Methods for each metric: "llm_judge", "rouge" (for ROUGE only), or "lettucedetect" (for hallucination)
    response_relevance_method: str = "llm_judge"
    response_completeness_method: str = "llm_judge"
    hallucination_method: str = "llm_judge"
    rouge_method: str = "rouge"
    
    # LettuceDetect-specific settings
    lettucedetect_model: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    
    # Weights for overall score calculation
    response_relevance_weight: float = 0.3
    response_completeness_weight: float = 0.3
    hallucination_weight: float = 0.3
    rouge_weight: float = 0.1
    
    # ROUGE-specific settings
    rouge_types: list[str] = None
    
    def __post_init__(self):
        if self.rouge_types is None:
            self.rouge_types = ["rouge-1", "rouge-2", "rouge-l"]


@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    include_detailed_explanations: bool = True
    include_context_sources: bool = True
    output_format: str = "both"  # text, json, both


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration"""
    metrics: MetricsConfig
    llm_provider: LLMProviderConfig
    report: ReportConfig


class ConfigLoader:
    """Loads and parses YAML configuration files"""
    
    @staticmethod
    def load(config_path: Optional[str] = None) -> EvaluationConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml in current directory.
        
        Returns:
            EvaluationConfig object
        """
        if config_path is None:
            config_path = "config.yaml"
        
        if not os.path.exists(config_path):
            # Return default config if file doesn't exist
            return ConfigLoader._default_config()
        
        if not YAML_AVAILABLE:
            print("Warning: PyYAML not installed. Using default configuration.")
            print("Install with: pip install pyyaml")
            return ConfigLoader._default_config()
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        # Parse metrics config
        metrics_data = config_data.get("metrics", {})
        metrics_config = MetricsConfig(
            response_relevance=metrics_data.get("response_relevance", {}).get("enabled", True),
            response_completeness=metrics_data.get("response_completeness", {}).get("enabled", True),
            hallucination=metrics_data.get("hallucination", {}).get("enabled", True),
            rouge=metrics_data.get("rouge", {}).get("enabled", True),
            response_relevance_method=metrics_data.get("response_relevance", {}).get("method", "llm_judge"),
            response_completeness_method=metrics_data.get("response_completeness", {}).get("method", "llm_judge"),
            hallucination_method=metrics_data.get("hallucination", {}).get("method", "llm_judge"),
            rouge_method=metrics_data.get("rouge", {}).get("method", "rouge"),
            lettucedetect_model=metrics_data.get("hallucination", {}).get("lettucedetect_model", "KRLabsOrg/lettucedect-base-modernbert-en-v1"),
            response_relevance_weight=metrics_data.get("response_relevance", {}).get("weight", 0.3),
            response_completeness_weight=metrics_data.get("response_completeness", {}).get("weight", 0.3),
            hallucination_weight=metrics_data.get("hallucination", {}).get("weight", 0.3),
            rouge_weight=metrics_data.get("rouge", {}).get("weight", 0.1),
            rouge_types=metrics_data.get("rouge", {}).get("rouge_types", ["rouge-1", "rouge-2", "rouge-l"])
        )
        
        # Parse LLM provider config
        llm_data = config_data.get("llm_provider", {})
        llm_config = LLMProviderConfig(
            provider=llm_data.get("provider", "openai"),
            model=llm_data.get("model", "gpt-4o-mini"),
            api_key=llm_data.get("api_key")
        )
        
        # Parse report config
        report_data = config_data.get("report", {})
        report_config = ReportConfig(
            include_detailed_explanations=report_data.get("include_detailed_explanations", True),
            include_context_sources=report_data.get("include_context_sources", True),
            output_format=report_data.get("output_format", "both")
        )
        
        return EvaluationConfig(
            metrics=metrics_config,
            llm_provider=llm_config,
            report=report_config
        )
    
    @staticmethod
    def _default_config() -> EvaluationConfig:
        """Returns default configuration with all metrics enabled"""
        return EvaluationConfig(
            metrics=MetricsConfig(
                response_relevance=True,
                response_completeness=True,
                hallucination=True,
                rouge=False  # Disable ROUGE by default if package not installed
            ),
            llm_provider=LLMProviderConfig(),
            report=ReportConfig()
        )

