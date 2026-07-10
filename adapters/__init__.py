"""
adapters  —  UAGF-XAI S6 Intake Layer
======================================
Translates upstream platform outputs (S4 / S5) into the typed
data-transfer objects consumed by the CBEP evidence planner.

Public API
----------
GovernanceContext   DTO for S4 (UAGF-GMM / CGSA) governance data
GovernanceAdapter   Converts CGSA canonical report JSON → GovernanceContext

AuditContext        DTO for S5 (UAGF-TAM / AAA) audit data
AuditAdapter        Converts S5 engagement state JSON  → AuditContext
"""

from .s4_governance_adapter import GovernanceAdapter, GovernanceContext
from .s5_audit_adapter import AuditAdapter, AuditContext

__all__ = [
    "GovernanceAdapter",
    "GovernanceContext",
    "AuditAdapter",
    "AuditContext",
]
