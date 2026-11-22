"""training-for-vea: a staged train -> eval -> judge pipeline for VEA experiments.

Wraps the vendored open-instruct (third_party/open-instruct) OlmoRL engine for the
training stage, a vLLM rollout generator for the eval stage, and LLM/verifier judges
for the judge stage — all driven by one YAML with an Inspect-style terminal UI.
"""

__version__ = "0.1.0"
