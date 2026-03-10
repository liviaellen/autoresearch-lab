"""
Program Generator — converts domain inputs into program.md.

Three modes:
    1. Template mode (questionnaire):
       python -m generator.program_generator --domain agriculture

    2. Config mode (JSON file):
       python -m generator.program_generator --domain agriculture --config config.json

    3. LLM mode (natural language conversation):
       python -m generator.program_generator --llm --model local
       python -m generator.program_generator --llm --model gpt4o --domain healthcare
       python -m generator.program_generator --llm --model claude-sonnet --description problem.txt
"""

import argparse
import json
import os
import sys

from .domain_questions import (
    interactive_questionnaire,
    list_domains,
    load_domain,
    get_constraints,
)
from .templates import render_program, build_inputs_from_answers


def generate_from_answers(domain_name: str, answers: dict) -> str:
    """Generate program.md content from domain name and answers dict."""
    domain_config = load_domain(domain_name)
    inputs = build_inputs_from_answers(answers, domain_config)
    return render_program(inputs)


def generate_from_config(domain_name: str, config_path: str) -> str:
    """Generate program.md from a JSON config file."""
    with open(config_path) as f:
        config = json.load(f)

    answers = config.get("answers", config)
    return generate_from_answers(domain_name, answers)


def generate_interactive(domain_name: str) -> str:
    """Run interactive questionnaire and generate program.md."""
    print(f"\n=== Autoresearch Program Generator ===")
    print(f"Domain: {domain_name}\n")

    answers = interactive_questionnaire(domain_name)

    print("\n--- Generating program.md ---\n")
    return generate_from_answers(domain_name, answers)


def main():
    parser = argparse.ArgumentParser(
        description="Generate program.md from domain-specific inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Modes:
  Template (default):  --domain agriculture
  Config file:         --domain agriculture --config config.json
  LLM conversation:    --llm --model local
  LLM single-shot:     --llm --model gpt4o --description problem.txt

LLM model presets: local, gpt4o, claude-sonnet, groq-llama, etc.
Run with --llm --list-models to see all presets.
""",
    )
    parser.add_argument(
        "--domain",
        help=f"Domain name. Available: {', '.join(list_domains())}",
    )
    parser.add_argument(
        "--config",
        help="Path to JSON config file (template mode)",
    )
    parser.add_argument(
        "--output",
        default="program.md",
        help="Output file path (default: program.md)",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available domains and exit",
    )
    # LLM mode arguments
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-powered generation instead of templates",
    )
    parser.add_argument(
        "--model",
        default="local",
        help="LLM model name or preset (default: local = ollama/llama3.2)",
    )
    parser.add_argument(
        "--description",
        help="Path to problem description file (LLM single-shot mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available LLM model presets and exit",
    )
    args = parser.parse_args()

    if args.list_domains:
        for d in list_domains():
            print(f"  {d}")
        return

    if args.list_models:
        from .llm_client import PRESETS
        print("\nAvailable model presets:\n")
        max_key = max(len(k) for k in PRESETS)
        for name, model in PRESETS.items():
            print(f"  {name:<{max_key + 2}} → {model}")
        print(f"\n  Or any LiteLLM model string (e.g. ollama/phi3, groq/mixtral-8x7b)\n")
        return

    # LLM mode
    if args.llm:
        from .llm_client import LLMConfig, resolve_model
        from .llm_generator import run_conversation, single_shot

        model = resolve_model(args.model)
        config = LLMConfig(model=model, temperature=args.temperature)

        if args.description:
            with open(args.description) as f:
                desc = f.read()
            content = single_shot(config, desc, domain=args.domain, output_path=args.output)
            print(f"Generated: {args.output} ({len(content)} chars)")
        else:
            run_conversation(config, domain=args.domain, output_path=args.output)
        return

    # Template mode (requires --domain)
    if not args.domain:
        parser.error("--domain is required for template mode. Use --llm for LLM mode.")

    if args.config:
        content = generate_from_config(args.domain, args.config)
    else:
        content = generate_interactive(args.domain)

    with open(args.output, "w") as f:
        f.write(content)

    print(f"Generated: {args.output}")
    print(f"Size: {len(content)} chars")


if __name__ == "__main__":
    main()
