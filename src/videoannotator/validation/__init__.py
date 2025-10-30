"""VideoAnnotator Configuration Validation.

Schema-based validation for pipeline and job configurations.
"""

from .models import FieldError, FieldWarning, ValidationResult
from .validator import ConfigValidator

__all__ = ["ConfigValidator", "FieldError", "FieldWarning", "ValidationResult"]
