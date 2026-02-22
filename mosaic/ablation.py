from enum import Enum


class Ablation(str, Enum):
    NONE = "none"
    NO_VERIFIER = "no_verifier"
    PDM_CLOSED_ONLY = "pdm_closed_only"
    FLOW_DRIVE_ONLY = "flow_drive_only"
