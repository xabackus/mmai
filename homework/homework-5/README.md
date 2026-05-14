# Homework 5: Agentic Multimodal AI

## Reading Assignment
Survey of three papers on agentic LLMs, covering the distinction between chatbots and agents, multi-step decision-making, and human-in-the-loop collaboration. Designed a hypothetical WCAG accessibility audit agent with formal observation/action spaces and evaluation criteria.

## Hands-On

### Accessibility Audit Agent (smolagents)
Built a WCAG web accessibility auditor using Hugging Face's smolagents framework:
- **Text-only agent** (Qwen2.5-7B, local): Custom `check_html_accessibility` and `wcag_guideline_lookup` tools. Found 37 issues on the W3C demo page but got stuck in code parsing loops.
- **Vision agent** (GPT-4o-mini, API): Added browser-based tools (helium) + screenshot capture. Identified visual accessibility issues (contrast, rendered layout) that HTML parsing alone misses. Completed audits in 3 steps vs 9.

### Safety Evaluation
Tested three adversarial prompts (credential entry, data scraping, prompt injection) before and after system prompt hardening. Key finding: prompt-level guardrails blocked 2/3 attacks, but a direct "ignore all previous instructions" injection defeated them in both conditions — structural mitigations (input classifiers, tool allowlists) are necessary.

### Agent Configurations
Compared CodeAgent vs ToolCallingAgent and different step budgets. ToolCallingAgent was the clear winner: zero code parsing errors, 4x richer output, competitive latency.

### Discord Deployment
Deployed the agent as a Discord bot with hybrid triggering (@mention + keyword detection). Successfully refused out-of-scope requests while responding to accessibility audit keywords.

### OpenClaw (Optional)
Explored OpenClaw as a persistent agent daemon. Compared its always-on architecture with smolagents' ephemeral design, discussing the security implications of persistent system-level access.

## Files
- `Xander_Backus_HW5_Written_Responses.pdf` — All parts (reading + hands-on + deployment + analysis)
