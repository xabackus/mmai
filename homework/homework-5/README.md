# Homework 5: Agentic Multimodal AI

## Overview

Built and evaluated an agentic WCAG web accessibility auditor using Hugging Face's smolagents framework. Covers the full lifecycle: reading assignment on agent architectures, tool design, baseline and custom-tool evaluation, vision-enhanced agents, safety testing, configuration benchmarking, observability with Langfuse, and Discord deployment.

## What I did

- **Reading assignment**: surveyed three papers on agentic LLMs. Designed a formal agent specification (observation/action spaces, transition dynamics, stopping conditions) for a WCAG accessibility auditor. Compared autonomous vs human-in-the-loop architectures.

- **Built a custom-tool agent** with `check_html_accessibility` and `wcag_guideline_lookup` tools. The custom tools eliminated the baseline's most critical failure — hallucinating violations on accessible pages — by programmatically verifying HTML structure instead of relying on the model's pattern-matching.

- **Vision-enhanced agent** (GPT-4o-mini + browser screenshots) identified visual accessibility issues (contrast, rendered layout) that HTML parsing alone cannot detect. Completed audits in 3 steps vs the text-only agent's 9.

- **Safety evaluation**: tested three adversarial prompts before and after system prompt hardening. Prompt injection ("ignore all previous instructions") defeated guardrails in both conditions, demonstrating that structural mitigations (input classifiers, tool allowlists) are necessary beyond prompt-level rules.

- **Agent configurations**: benchmarked CodeAgent vs ToolCallingAgent across step budgets. ToolCallingAgent won on reliability (zero parsing errors), output quality (4x richer reports), and competitive latency.

- **Discord deployment**: built a bot with hybrid trigger (@mention + keyword detection) that correctly refuses out-of-scope requests while responding to accessibility audit queries.

## Connection to the final project

The connection is less direct than HW1–4, but there's a shared theme: building systems that respond appropriately to the *type* of input they receive. In this homework, the agent must distinguish audit requests from adversarial prompts and refuse the latter. In EgoBlind-RA, the system must distinguish urgent from non-urgent queries and respond differently to each. Both require a classification stage that gates downstream behavior — the CLIP urgency classifier in our project serves an analogous role to the safety classifier / scope detection in the agent. The safety evaluation findings (prompt-level guardrails are insufficient; you need structural separation) also rhyme with our paper's finding that prompt-level urgency conditioning (Approach 2) and structural urgency separation (Approach 1) have different robustness properties.

## Files

- `Xander_Backus_HW5_Written_Responses.pdf` — All written responses (parts 1–6 + optional)
- `Homework_5_AI_Agents.ipynb` — Agent notebook (smolagents, Langfuse, safety eval)
