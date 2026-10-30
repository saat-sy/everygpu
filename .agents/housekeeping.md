# Housekeeping

Rules for application work in this repo. Read this and `context.md` first. Messaging and session administration use the shared ping skill without application-context reads or an application plan.

## Files

- Small, concise, to the point. No filler, no restating, no slop.
- State the current design only. Edit in place; no "previously" or obsolete options.

## Plans

- A plan is a branch. Working on branch `X` means `.agents/plan/X_plan.md` exists.
- Write the plan after the user gives requirements and you have clarified doubts. No code before the plan.
- Plan holds: goal, requirements, steps, decisions, and a `Learnings` section.
- Before starting a plan, read prior plans' `Learnings`, the current plan's `Learnings` and this file. Do not repeat a recorded mistake.

## Codex collaboration

- Before coding or delegation, show the concrete workflow: commands, inputs, outputs, defaults and completion criteria. Ask about unresolved product choices; do not turn proposals into requirements.
- After agreement, carry routine implementation, review fixes and authorized handoffs forward without another prompt. Ask before changing user-facing scope; wait for approval at numbered plan boundaries.
- Codex plans and audits; Claude implements when delegation is authorized. Give a fresh implementer session a self-contained handoff with agreed behavior, constraints and verification. Check separate user instructions before challenging another agent's changes.
- Apply YAGNI strongly: keep only what the agreed workflow needs. Prefer clear responsibilities and flow over generic helpers or extra layers. Do not invent commands, argument limits, files, abstractions or standing rules. Choose sensible defaults for mechanical details; make consequential behavior explicit.
- Report fixture results, real-checkpoint results and remaining work separately. Label unmeasured resource and timing estimates; inspect tensor sizes and code memory usage before rental advice. Keep individual feedback private when requested.

## Distill

On `distill`:

1. Add or update `Learnings` in the branch plan: mistakes, corrections, what worked.
2. Update `context.md` to match what changed.
3. Put the why behind architecture decisions in `docs/<topic>.md`, one decision per file, so no one re-derives it from the code.
4. Promote any new standing rule to this file.

## Python

- Format with `black` (dev dependency group `dev`) before finishing.
- Docstrings, comments and help text are single lines. Never multi-line.
- Every `__init__.py` stays empty.
- Group code in feature subpackages with a shared `utils/` (constants, models, helpers); keep the package top level to the entrypoint.
- One responsibility per file; more files are fine when each is independent. Name operations explicitly and keep helpers private next to the code that uses them.
- Each component installs from its own dependency group in the root `pyproject.toml`.

## Environment

- Do not download models or run real-model, GPU or upload work in this checkout. Use small fixtures and mocks, and ask first.

## Tests

- Do not write tests unless the user asks.

## Git

- Never commit or change history unless asked.
