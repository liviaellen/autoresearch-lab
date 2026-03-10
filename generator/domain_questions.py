"""
Domain-specific question engine.

Each domain defines a set of questions that guide the user through
describing their research problem. Answers become structured inputs
for program_generator.py.
"""

import os
import yaml


DOMAINS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "domains")


def load_domain(domain_name: str) -> dict:
    """Load a domain definition from YAML."""
    path = os.path.join(DOMAINS_DIR, f"{domain_name}.yaml")
    if not os.path.exists(path):
        available = list_domains()
        raise FileNotFoundError(
            f"Domain '{domain_name}' not found at {path}. "
            f"Available domains: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def list_domains() -> list[str]:
    """List all available domain names."""
    if not os.path.isdir(DOMAINS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(DOMAINS_DIR)
        if f.endswith(".yaml")
    )


def get_questions(domain_name: str) -> list[dict]:
    """Get the question list for a domain.

    Each question is a dict with keys:
        - id: unique identifier
        - prompt: the question text shown to the user
        - type: "text", "choice", or "number"
        - choices: list of options (only for type="choice")
        - default: optional default value
        - required: bool
    """
    domain = load_domain(domain_name)
    return domain.get("questions", [])


def get_constraints(domain_name: str) -> list[str]:
    """Get domain-specific constraints."""
    domain = load_domain(domain_name)
    return domain.get("constraints", [])


def get_defaults(domain_name: str) -> dict:
    """Get domain default values."""
    domain = load_domain(domain_name)
    return domain.get("defaults", {})


def interactive_questionnaire(domain_name: str) -> dict:
    """Run an interactive CLI questionnaire for a domain.

    Returns a dict of {question_id: answer}.
    """
    questions = get_questions(domain_name)
    defaults = get_defaults(domain_name)
    answers = {}

    for q in questions:
        qid = q["id"]
        prompt = q["prompt"]
        qtype = q.get("type", "text")
        default = q.get("default") or defaults.get(qid)
        required = q.get("required", True)
        choices = q.get("choices", [])

        if default:
            prompt_str = f"{prompt} [{default}]: "
        else:
            prompt_str = f"{prompt}: "

        if choices:
            print(f"\n{prompt}")
            for i, choice in enumerate(choices, 1):
                print(f"  {i}. {choice}")
            if default:
                prompt_str = f"Choose (1-{len(choices)}) [{default}]: "
            else:
                prompt_str = f"Choose (1-{len(choices)}): "

        while True:
            raw = input(prompt_str).strip()
            if not raw and default:
                raw = str(default)
            if not raw and required:
                print("  This field is required.")
                continue

            if choices:
                try:
                    idx = int(raw) - 1
                    if 0 <= idx < len(choices):
                        answers[qid] = choices[idx]
                        break
                    else:
                        print(f"  Please choose 1-{len(choices)}")
                except ValueError:
                    if raw in choices:
                        answers[qid] = raw
                        break
                    print(f"  Please choose 1-{len(choices)}")
            elif qtype == "number":
                try:
                    answers[qid] = float(raw) if "." in raw else int(raw)
                    break
                except ValueError:
                    print("  Please enter a number.")
            else:
                answers[qid] = raw
                break

    return answers
