"""
Experiments Module - Scientific Experiment Framework

CRITICAL: Every new method MUST be an EXPERIMENT, not a "feature".

Each experiment requires:
1. HYPOTHESIS - What do we expect to prove/disprove?
2. PARAMETERS - What inputs/thresholds affect the outcome?
3. EXPECTED EFFECT - How will we measure success?

Experiments that pass validation can graduate to production modules.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    HYPOTHESIS = "hypothesis"      # Just an idea, not implemented
    IMPLEMENTING = "implementing"  # Being coded
    TESTING = "testing"            # Under validation
    VALIDATED = "validated"        # Passed tests, ready for production
    REJECTED = "rejected"          # Failed validation
    GRADUATED = "graduated"        # Moved to production module


@dataclass
class Experiment:
    """
    Scientific experiment definition.
    
    Every new method in this codebase MUST be defined as an experiment
    before becoming a production feature.
    """
    # Required fields
    name: str                          # Unique identifier
    hypothesis: str                    # What we expect to prove
    expected_effect: str               # How we measure success
    
    # Parameters that affect the experiment
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_descriptions: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    status: ExperimentStatus = ExperimentStatus.HYPOTHESIS
    created_date: str = field(default_factory=lambda: datetime.now().isoformat()[:10])
    author: str = ""
    
    # Validation results
    validation_notes: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    
    # Reference to implementation
    implementation_module: Optional[str] = None
    implementation_function: Optional[str] = None
    
    def validate(self) -> bool:
        """Check if experiment definition is complete."""
        return bool(
            self.name and
            self.hypothesis and
            self.expected_effect and
            len(self.success_criteria) > 0
        )
    
    def to_markdown(self) -> str:
        """Generate markdown documentation for this experiment."""
        params_md = "\n".join([
            f"  - `{k}`: {self.parameter_descriptions.get(k, 'No description')} (default: {v})"
            for k, v in self.parameters.items()
        ]) or "  - None"
        
        criteria_md = "\n".join([f"  - {c}" for c in self.success_criteria]) or "  - TBD"
        
        return f"""## {self.name}

**Status**: {self.status.value}

### Hypothesis
{self.hypothesis}

### Parameters
{params_md}

### Expected Effect
{self.expected_effect}

### Success Criteria
{criteria_md}

### Implementation
- Module: `{self.implementation_module or 'TBD'}`
- Function: `{self.implementation_function or 'TBD'}`
"""


# ============================================================
# Experiment Registry
# ============================================================

_EXPERIMENTS: Dict[str, Experiment] = {}


def register_experiment(experiment: Experiment) -> None:
    """Register an experiment in the global registry."""
    if not experiment.validate():
        raise ValueError(f"Invalid experiment: {experiment.name} - missing required fields")
    _EXPERIMENTS[experiment.name] = experiment


def get_experiment(name: str) -> Optional[Experiment]:
    """Get an experiment by name."""
    return _EXPERIMENTS.get(name)


def list_experiments(status: Optional[ExperimentStatus] = None) -> List[Experiment]:
    """List all experiments, optionally filtered by status."""
    experiments = list(_EXPERIMENTS.values())
    if status:
        experiments = [e for e in experiments if e.status == status]
    return experiments


def generate_experiments_report() -> str:
    """Generate a markdown report of all experiments."""
    if not _EXPERIMENTS:
        return "# Experiments\n\nNo experiments registered."
    
    lines = ["# Experiments Registry\n"]
    
    for status in ExperimentStatus:
        exps = list_experiments(status)
        if exps:
            lines.append(f"\n## {status.value.title()} ({len(exps)})\n")
            for exp in exps:
                lines.append(exp.to_markdown())
    
    return "\n".join(lines)


# ============================================================
# Example Experiments
# ============================================================

# Example: DFA Threshold Detection
DFA_THRESHOLD_EXPERIMENT = Experiment(
    name="dfa_threshold_detection",
    hypothesis="DFA-a1 crossing 0.75 correlates with VT1, crossing 0.5 correlates with VT2",
    expected_effect="Improved threshold detection accuracy compared to VE-only method",
    parameters={
        "vt1_alpha_threshold": 0.75,
        "vt2_alpha_threshold": 0.50,
        "min_window_sec": 120,
    },
    parameter_descriptions={
        "vt1_alpha_threshold": "Alpha1 value at VT1 transition",
        "vt2_alpha_threshold": "Alpha1 value at VT2 transition",
        "min_window_sec": "Minimum window for reliable DFA calculation",
    },
    status=ExperimentStatus.TESTING,
    author="research",
    success_criteria=[
        "VT1 detected within ±10W of VE-based VT1",
        "VT2 detected within ±15W of VE-based VT2",
        "False positive rate < 10%",
    ],
    failure_modes=[
        "Artifacts in RR data cause DFA > 1.5 at high intensity",
        "Short recording (<10 min) insufficient for stable α1",
    ],
    implementation_module="modules.calculations.hrv",
    implementation_function="calculate_dynamic_dfa",
)

# Example: SmO2 Local Signal Limitation
SMO2_LOCAL_SIGNAL_EXPERIMENT = Experiment(
    name="smo2_local_signal_limitation",
    hypothesis="SmO2 thresholds differ from VE thresholds by >15W due to local vs systemic measurement",
    expected_effect="Reduce overconfidence in SmO2-only threshold detection",
    parameters={
        "max_agreement_threshold_watts": 15,
        "confidence_reduction_factor": 0.5,
    },
    parameter_descriptions={
        "max_agreement_threshold_watts": "Max W difference to consider SmO2 as confirming VT",
        "confidence_reduction_factor": "Factor to reduce confidence when SmO2 disagrees with VT",
    },
    status=ExperimentStatus.VALIDATED,
    author="research",
    success_criteria=[
        "SmO2 marked as LOCAL signal in all outputs",
        "SmO2 thresholds used only as supporting evidence",
        "User warnings displayed when SmO2 disagrees with VT",
    ],
    failure_modes=[
        "User incorrectly uses SmO2 thresholds as primary decision",
    ],
    implementation_module="modules.calculations.metabolic",
    implementation_function="detect_smo2_from_steps",
)

# Register example experiments
register_experiment(DFA_THRESHOLD_EXPERIMENT)
register_experiment(SMO2_LOCAL_SIGNAL_EXPERIMENT)


__all__ = [
    # Enums
    'ExperimentStatus',
    # Classes
    'Experiment',
    # Functions
    'register_experiment',
    'get_experiment',
    'list_experiments',
    'generate_experiments_report',
    # Example experiments
    'DFA_THRESHOLD_EXPERIMENT',
    'SMO2_LOCAL_SIGNAL_EXPERIMENT',
]
