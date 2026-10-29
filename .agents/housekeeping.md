# Housekeeping

Rules for application work in this repo. Read this and `context.md` first. Messaging and session administration use the shared ping skill without application-context reads or an application plan.

## Files

- Small, concise, to the point. No filler, no restating, no slop.
- State the current design only. Edit in place; no "previously" or obsolete options.

## Plans

- A plan is a branch. Working on branch `X` means `.agents/plan/X_plan.md` exists.
- Write the plan after the user gives requirements and you have clarified doubts. No code before the plan.
- Plan holds: goal, requirements, steps, decisions, and a `Learnings` section.
- Read the plan's `Learnings` and this file before implementing. Do not repeat a recorded mistake.

## Distill

On `distill`:

1. Add or update `Learnings` in the branch plan: mistakes, corrections, what worked.
2. Update `context.md` to match what changed.
3. Put the why behind architecture decisions in `docs/<topic>.md`, one decision per file, so no one re-derives it from the code.
4. Promote any new standing rule to this file.

## Formatting

- Format all Python with `black` (dev dependency group `dev`) before finishing.

## Comments

- Docstrings, comments and help text are single lines. Never multi-line.

## Tests

- Do not write tests unless the user asks.

## Git

- Never commit or change history unless asked.
