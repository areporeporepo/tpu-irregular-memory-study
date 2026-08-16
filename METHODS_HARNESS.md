# Methods: the harness, and whether it is the right loop

This study is run by an agent. That is worth documenting precisely, because in 2026 the harness is
a research object rather than an implementation detail, and because the harness caught more errors
than the human or either model did.

## What the literature calls this

The 2026 framing has a name: **harness engineering**. Recent work (*Harness Engineering for
Language Agents: The Harness Layer as Control, Agency, and Runtime*, and *LongHorizon-Harness*)
argues that long-horizon agency is not a property of the model but of the externalised layer around
it: loops and workflows, context and memory, tools and skills, orchestration, hooks, and
verification. Adjacent threads: **ReAcTree** distributes long-horizon planning across recursive
sub-agents with explicit control flow; **CaveAgent** makes a persistent runtime the locus of agent
state; **AgentOrchestra** and **AOrchestra** automate sub-agent creation; **SemanticALLI** caches
intermediate reasoning rather than responses. The consensus is that nobody uses pure ReAct or pure
graph execution any more, and production systems route between them.

Against that taxonomy, this campaign is a **supervisor-worker loop with an external verifier and an
adversarial reviewer**, where the workers are shell commands on preemptible accelerators rather
than sub-agents.

## The actual loop

```
launchd, every 20 minutes
  └── supervisor.sh                      # no model in this path, deliberately
        ├── budget_guard.py              # bill the fleet, STOP or THROTTLE if the reserve is at risk
        ├── reclaim capacity             # spot preemption is expected, not exceptional
        ├── run experiments              # bounded, restartable, one cycle loses at most one cycle
        ├── observe_contention.py        # the shared class project, hashed
        ├── logbook.py build             # regenerate the page from JSON
        └── git push                     # publish, unattended

Claude Code (Opus 5), in the outer loop
  ├── designs the next experiment from the last cycle's data
  ├── writes it, ships it, reads the result
  ├── logs findings, corrections, decisions and plan changes as JSON lines
  └── consults Gemini 3.1 Pro as an adversarial reviewer at each turning point
```

The important design choice: **the recurring path contains no model call.** A 20-minute cycle that
needed an LLM would be fragile, expensive and unauditable. The model does design and interpretation;
the shell does collection. That split is why the campaign survives being ignored for hours.

## Which harness features earned their place

| Feature | What it bought |
|---|---|
| **Background tasks** | Six TPU jobs in flight at once: a 256-chip burst provisioning while a 32-chip sweep measured while a single chip probed kernel shapes. Serial execution would have wasted the expensive slice's clock. |
| **launchd, not an agent loop** | Collection continues with nobody watching. The 20-minute cycle has run unattended repeatedly, including through a preemption. |
| **A hook that blocked me** | The dangerous-command hook refused two of my commands, one for a pattern that looked like a force-push. Annoying twice, correct once. |
| **1M context** | The whole campaign is one session: every measurement, correction and dead end is still addressable, which is why the 16-chip discrepancy across two slice sizes was noticed at all. It appeared in runs hours apart. |
| **A second model** | Gemini 3.1 Pro changed the direction twice, documented below. |
| **Generated logbook** | 23 entries as JSON lines rendering to a page that republishes itself. Nothing is hand-maintained, so nothing is stale. |

## What the harness caught that the model did not

This is the part worth reporting, because it is evidence about harness value rather than opinion.

**verify.py caught three method failures** that would have gone into a writeup:
1. Per-op timing moved 2.44x with chain length, so every quoted latency was an overestimate.
2. A sub-mesh timed a 16 MiB all_to_all at 0.0042 ms, implying 3.5 TB/s per chip. Impossible.
3. The gather harness was non-monotonic: dim=32 slower than dim=8.

**Gemini caught two conceptual errors:**
1. It identified SparseCore as a possibly deprecating branch, which reframed the research question
   from "benchmark SparseCore" to "where does irregular work belong".
2. It replaced my host-boundary explanation of a +77% cost step with bisection congestion. A later
   256-chip run confirmed Gemini and refuted me: identical 16 chips cost 1.76x differently depending
   on the parent slice.

**Neither model caught the third thing.** The topology dump did: `contiguous` and `reversed` are
both compact 4x2 blocks with identical diameter 4, so a 56x measured gap could not be geometry.
That came from printing device coordinates rather than from reasoning.

## Is this the best loop? Probably not, and here is what is missing

Honest assessment against the 2026 taxonomy:

- **No automated hypothesis generation.** Gemini proposes hypotheses when asked. The loop does not
  ask on its own. Discovery Loop, the Dean, Ghemawat, Vinyals and Le company launched 5 August 2026,
  is built precisely to close that: propose, run, and learn from thousands of parallel evaluations.
  Ours runs one experiment at a time because a human reads each result.
- **No verifier in the recurring path.** `verify.py` runs when invoked. It should gate every cycle,
  refusing to publish numbers that fail their own checks. That is the single highest-value fix.
- **No memory beyond the log.** Context is one long session. When it ends, continuity depends on
  the logbook being good enough to reconstruct intent, which is an argument for writing the log
  for a stranger.
- **Sub-agents unused.** Six parallel background shells, but no parallel *reasoning*. Independent
  analysis of the banked HLO would parallelise cleanly.

The honest summary: this is a good measurement harness and a mediocre discovery harness. It
compresses the run-and-collect loop to minutes and leaves the propose-and-interpret loop at human
speed. That is the gap the field is currently racing to close, and it is visible from inside.
