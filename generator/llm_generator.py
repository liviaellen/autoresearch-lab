"""
LLM-powered program.md generator.

Instead of rigid questionnaires, the user describes their research problem
in natural language. An LLM (any provider) conducts a guided conversation,
then generates a structured program.md.

Usage:
    # Interactive conversation (default: local ollama)
    python -m generator.llm_generator

    # With specific model
    python -m generator.llm_generator --model gpt4o
    python -m generator.llm_generator --model claude-sonnet
    python -m generator.llm_generator --model ollama/llama3.2

    # Single-shot from description file
    python -m generator.llm_generator --description problem.txt --model local

    # List available presets
    python -m generator.llm_generator --list-models
"""

import argparse
import json
import os
import sys

from .llm_client import LLMConfig, chat, chat_stream, resolve_model, PRESETS
from .domain_questions import list_domains, load_domain
from .templates import PROGRAM_TEMPLATE


SYSTEM_PROMPT = """\
You are an autonomous research program designer. Your job is to help domain experts \
set up AI-driven experiments by generating a structured program.md file.

You will:
1. Understand the user's research problem through conversation
2. Ask clarifying questions (2-4 max) to fill gaps
3. Generate a complete, structured program.md

When you have enough information, output the final program.md between \
<program> and </program> tags.

The program.md MUST include these sections:
- Research Objective: Clear problem statement
- Dataset: Schema, location, size, any known issues
- Evaluation Metric: Primary metric and direction (lower/higher is better)
- Experiment Strategy: What models to try, in what order, what to focus on
- Allowed Modifications: Which files the agent can edit (train.py and/or train_mlx.py)
- Forbidden Actions: What the agent must NOT do
- Constraints: Domain-specific requirements (interpretability, fairness, etc.)
- Runtime: Time budget per experiment
- Reporting: How results should be logged

Be specific and practical. The output will be used by an autonomous AI agent \
to run experiments without human supervision.

{domain_context}
"""

DOMAIN_CONTEXT_TEMPLATE = """\
The user is working in the {name} domain.

Domain-specific constraints to consider:
{constraints}

Recommended model families for this domain:
{model_families}

Key questions to consider for this domain:
{questions}
"""


def build_system_prompt(domain_name: str | None = None) -> str:
    """Build system prompt, optionally enriched with domain context."""
    domain_context = ""

    if domain_name:
        try:
            domain = load_domain(domain_name)
            constraints = "\n".join(f"- {c}" for c in domain.get("constraints", []))
            defaults = domain.get("template_defaults", {})
            models = "\n".join(f"- {m}" for m in defaults.get("model_families", []))
            questions = "\n".join(
                f"- {q['prompt']}" for q in domain.get("questions", [])
            )
            domain_context = DOMAIN_CONTEXT_TEMPLATE.format(
                name=domain.get("name", domain_name),
                constraints=constraints or "None specified",
                model_families=models or "Any",
                questions=questions or "None specified",
            )
        except FileNotFoundError:
            domain_context = f"Domain '{domain_name}' requested but no config found. Infer from conversation."

    return SYSTEM_PROMPT.format(domain_context=domain_context)


def extract_program(text: str) -> str | None:
    """Extract program.md content from between <program> tags."""
    start = text.find("<program>")
    end = text.find("</program>")
    if start != -1 and end != -1:
        return text[start + len("<program>"):end].strip()
    return None


def run_conversation(
    config: LLMConfig,
    domain: str | None = None,
    initial_description: str | None = None,
    output_path: str = "program.md",
    stream: bool = True,
):
    """Run an interactive conversation to generate program.md.

    Args:
        config: LLM configuration (model, temperature, etc.)
        domain: Optional domain name to load context from YAML.
        initial_description: If provided, skip the first user prompt.
        output_path: Where to write the generated program.md.
        stream: Whether to stream LLM responses.
    """
    system_prompt = build_system_prompt(domain)
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n{'='*60}")
    print(f"  Autoresearch Program Generator (LLM-powered)")
    print(f"  Model: {config.model}")
    if domain:
        print(f"  Domain: {domain}")
    print(f"{'='*60}")
    print()
    print("Describe your research problem. Include:")
    print("  - What you're trying to predict/classify")
    print("  - What data you have")
    print("  - Any constraints (interpretability, fairness, etc.)")
    print()
    print("Type 'done' to finish, 'quit' to exit without saving.")
    print(f"{'─'*60}\n")

    # If we have an initial description, use it as the first message
    if initial_description:
        print(f"[Loading description...]\n")
        messages.append({"role": "user", "content": initial_description})
    else:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            print("Exiting.")
            return
        messages.append({"role": "user", "content": user_input})

    program_content = None
    max_turns = 10

    for turn in range(max_turns):
        # Get LLM response
        print("\nAssistant: ", end="", flush=True)

        if stream:
            full_response = ""
            for chunk in chat_stream(messages, config):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
        else:
            full_response = chat(messages, config)
            print(full_response)

        messages.append({"role": "assistant", "content": full_response})

        # Check if the LLM generated the program
        program_content = extract_program(full_response)
        if program_content:
            break

        # Get next user input
        print()
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Exiting without saving.")
            return

        if user_input.lower() == "done":
            # Ask the LLM to generate the final program
            messages.append({
                "role": "user",
                "content": "Please generate the final program.md now based on our conversation. "
                           "Output it between <program> and </program> tags.",
            })
            continue

        messages.append({"role": "user", "content": user_input})

    if program_content is None:
        # Final attempt: ask explicitly
        messages.append({
            "role": "user",
            "content": "Generate the program.md now. Output between <program> and </program> tags.",
        })
        final = chat(messages, config)
        program_content = extract_program(final)

    if program_content:
        with open(output_path, "w") as f:
            f.write(program_content + "\n")
        print(f"\n{'─'*60}")
        print(f"Generated: {output_path}")
        print(f"Size: {len(program_content)} chars")
        print(f"{'─'*60}")

        # Also save the conversation for reproducibility
        conv_path = output_path.replace(".md", "_conversation.json")
        with open(conv_path, "w") as f:
            json.dump(messages, f, indent=2)
        print(f"Conversation saved: {conv_path}")
    else:
        print("\nFailed to generate program.md. Try providing more detail or a different model.")


def single_shot(
    config: LLMConfig,
    description: str,
    domain: str | None = None,
    output_path: str = "program.md",
) -> str:
    """Generate program.md in a single LLM call (no conversation).

    Good for automation and CI pipelines.

    Args:
        config: LLM configuration.
        description: Full problem description.
        domain: Optional domain name.
        output_path: Where to write output.

    Returns:
        The generated program.md content.
    """
    system_prompt = build_system_prompt(domain)
    prompt = (
        f"Generate a complete program.md for the following research problem. "
        f"Output it between <program> and </program> tags.\n\n"
        f"Problem description:\n{description}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = chat(messages, config)
    program_content = extract_program(response)

    if program_content:
        with open(output_path, "w") as f:
            f.write(program_content + "\n")
        return program_content

    raise RuntimeError("LLM did not produce a valid program.md. Raw response saved to stderr.")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered program.md generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Interactive with local llama
  python -m generator.llm_generator --model local

  # Interactive with GPT-4o
  python -m generator.llm_generator --model gpt4o

  # Interactive with Claude
  python -m generator.llm_generator --model claude-sonnet

  # Single-shot from file
  python -m generator.llm_generator --model local --description problem.txt

  # With domain context
  python -m generator.llm_generator --model local --domain agriculture

  # Custom Ollama model
  python -m generator.llm_generator --model ollama/deepseek-r1

  # Any OpenAI-compatible API
  python -m generator.llm_generator --model openai/my-model --base-url http://localhost:8000/v1
""",
    )
    parser.add_argument(
        "--model",
        default="local",
        help="Model name or preset (default: local). Use --list-models to see presets.",
    )
    parser.add_argument(
        "--domain",
        help=f"Domain context. Available: {', '.join(list_domains())}",
    )
    parser.add_argument(
        "--description",
        help="Path to a text file with the problem description (single-shot mode).",
    )
    parser.add_argument(
        "--output",
        default="program.md",
        help="Output file path (default: program.md)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens (default: 4096)",
    )
    parser.add_argument(
        "--base-url",
        help="Custom API base URL (for OpenAI-compatible servers)",
    )
    parser.add_argument(
        "--api-key",
        help="API key (or set via env var for your provider)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (wait for full response)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model presets and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable model presets:\n")
        max_key = max(len(k) for k in PRESETS)
        for name, model in PRESETS.items():
            print(f"  {name:<{max_key + 2}} → {model}")
        print(f"\n  Or use any LiteLLM model string directly (e.g. ollama/phi3, groq/mixtral-8x7b)")
        print(f"  Full list: https://docs.litellm.ai/docs/providers\n")
        return

    model = resolve_model(args.model)
    config = LLMConfig(
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.description:
        with open(args.description) as f:
            description = f.read()
        content = single_shot(config, description, domain=args.domain, output_path=args.output)
        print(f"Generated: {args.output} ({len(content)} chars)")
    else:
        run_conversation(
            config,
            domain=args.domain,
            output_path=args.output,
            stream=not args.no_stream,
        )


if __name__ == "__main__":
    main()
