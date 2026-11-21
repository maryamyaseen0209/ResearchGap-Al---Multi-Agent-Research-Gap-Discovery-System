"""Phase 8: transparent, reproducible, human-auditable evaluations."""

from .baseline import SinglePassBaseline, BaselineResult
from .metrics import audit_workflow, compare_systems, create_review_template

__all__ = ["SinglePassBaseline", "BaselineResult", "audit_workflow", "compare_systems", "create_review_template"]
