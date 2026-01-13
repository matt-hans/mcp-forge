---
description: Autonomous mode - reads state, executes next step, spawns next Claude.
allowed-tools:
  - Read
  - Bash
  - Grep
  - Glob
  - SlashCommand
---

<objective>
Autonomous execution with self-spawning. Run once:
```bash
claude --dangerously-skip-permissions '/gsd:auto'
```

Reads STATE.md, determines next action, executes it, then spawns a fresh Claude to continue.
Chains until milestone complete. All gates auto-approved.
</objective>

<process>

<step name="verify_yolo">
**Check yolo mode is enabled:**

```bash
cat .planning/config.json 2>/dev/null
```

Parse the JSON. If `mode` is not `"yolo"`, refuse:

```
⚠ Auto mode requires yolo mode.

Set in .planning/config.json:
{"mode": "yolo"}

Then run:
while true; do claude -p --dangerously-skip-permissions '/gsd:auto'; done
```

Exit without executing anything.

If no `.planning/` directory exists:

```
No planning structure found.
Run /gsd:new-project first.
```

Exit.
</step>

<step name="load_state">
**Load minimal state:**

- Read `.planning/STATE.md` - current position
- Read `.planning/ROADMAP.md` - phase structure

Extract:
- Current phase number
- Current plan number (or "none")
- Phase status (planning/executing/complete)
- Total phases in milestone
</step>

<step name="determine_action">
**Determine next action based on state:**

Check what exists in `.planning/phases/`:

1. **No PLAN.md for current phase** → need to plan
2. **PLAN.md exists, no SUMMARY.md** → need to execute
3. **SUMMARY.md exists for all plans in phase** → phase complete, check next
4. **All phases complete** → milestone complete

Logic:
```
if current_phase has no PLAN.md:
  action = plan-phase
elif current_phase has unexecuted PLAN.md:
  action = execute-plan
elif current_phase complete AND more phases exist:
  action = plan-phase (next phase)
elif all phases complete:
  action = complete-milestone
```
</step>

<step name="execute">
**Execute the determined action immediately:**

No status report. No "Next Up" display. Just execute.

**If action = plan-phase:**
```
▶ AUTO: /gsd:plan-phase [N]
```
Invoke: `SlashCommand("/gsd:plan-phase [N]")`

**If action = execute-plan:**
```
▶ AUTO: /gsd:execute-plan [path]
```
Invoke: `SlashCommand("/gsd:execute-plan [full-path-to-PLAN.md]")`

**If action = complete-milestone:**
```
▶ AUTO: /gsd:complete-milestone
```
Invoke: `SlashCommand("/gsd:complete-milestone")`

**If milestone already complete:**
```
✅ Milestone complete. Nothing to do.
```
Exit cleanly (loop will restart, hit this again, user can Ctrl+C).
</step>

</process>

<auto_approve_context>
**CRITICAL: ALL HUMAN CHECKPOINTS ARE AUTO-APPROVED IN /gsd:auto**

When executing via /gsd:auto, treat ALL checkpoint types as auto-approved:

- `checkpoint:human-verify` → AUTO-APPROVED, log "Auto-verified in yolo mode"
- `checkpoint:decision` → AUTO-APPROVED, pick the most sensible default
- `checkpoint:human-action` → SKIP if possible, or simulate completion
- Any "Type approved to continue" → AUTO-APPROVED
- Any "Does this look right?" → YES
- Any confirmation prompt → CONFIRMED

**DO NOT STOP AND WAIT FOR INPUT. EVER.**

If a checkpoint would normally block, instead:
1. Log: "⚡ Auto-approved: [checkpoint description]"
2. Continue execution immediately

The invoked commands should detect yolo mode from config.json and apply this behavior.
</auto_approve_context>

<success_criteria>
- [ ] Refuses to run if mode != yolo
- [ ] Reads state, determines action
- [ ] Executes exactly one action via SlashCommand
- [ ] No prompts, no waiting, no "Next Up" display
- [ ] Exits cleanly after action completes
- [ ] Works with: while true; do claude -p --dangerously-skip-permissions '/gsd:auto'; done
</success_criteria>
