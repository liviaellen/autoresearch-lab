"""
Program template engine.

Converts structured inputs (domain answers + constraints) into
a standardized program.md file.
"""

from jinja2 import Environment, BaseLoader

PROGRAM_TEMPLATE = """\
# Research Objective

{{ objective }}

# Dataset

{{ dataset_description }}

{% if dataset_path -%}
**Location:** `{{ dataset_path }}`
{% endif %}

# Evaluation Metric

**Primary metric:** {{ metric }}
{% if metric_direction -%}
**Direction:** {{ metric_direction }}
{% endif %}

# Experiment Strategy

{{ strategy }}

{% if model_families -%}
**Model families to explore:**
{% for model in model_families -%}
- {{ model }}
{% endfor %}
{% endif %}

# Allowed Modifications

{% for file in allowed_files -%}
- `{{ file }}`
{% endfor %}

# Forbidden Actions

{% for action in forbidden_actions -%}
- {{ action }}
{% endfor %}

# Constraints

{% for constraint in constraints -%}
- {{ constraint }}
{% endfor %}

# Runtime

**Time budget per experiment:** {{ time_budget }}
{% if total_budget -%}
**Total research budget:** {{ total_budget }}
{% endif %}

# Reporting

{{ reporting }}
"""


def render_program(inputs: dict) -> str:
    """Render a program.md from structured inputs.

    Expected keys:
        - objective: str
        - dataset_description: str
        - dataset_path: str (optional)
        - metric: str
        - metric_direction: str (optional, e.g. "lower is better")
        - strategy: str
        - model_families: list[str] (optional)
        - allowed_files: list[str]
        - forbidden_actions: list[str]
        - constraints: list[str]
        - time_budget: str
        - total_budget: str (optional)
        - reporting: str
    """
    defaults = {
        "dataset_path": "",
        "metric_direction": "",
        "model_families": [],
        "allowed_files": ["train.py"],
        "forbidden_actions": [
            "Installing new packages",
            "Modifying evaluation pipeline",
            "Modifying data loading",
        ],
        "constraints": [],
        "time_budget": "5 minutes",
        "total_budget": "",
        "reporting": "Log results to results.tsv (tab-separated).",
    }

    merged = {**defaults, **inputs}
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(PROGRAM_TEMPLATE)
    return template.render(**merged)


def build_inputs_from_answers(answers: dict, domain_config: dict) -> dict:
    """Convert questionnaire answers + domain config into template inputs.

    Maps domain-specific answer keys to the standardized template keys.
    """
    mapping = domain_config.get("answer_mapping", {})
    constraints = domain_config.get("constraints", [])
    defaults = domain_config.get("template_defaults", {})

    inputs = dict(defaults)

    for answer_key, template_key in mapping.items():
        if answer_key in answers:
            inputs[template_key] = answers[answer_key]

    if constraints:
        existing = inputs.get("constraints", [])
        inputs["constraints"] = existing + constraints

    return inputs
