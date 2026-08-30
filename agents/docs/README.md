# `agents/docs` — long-lived reference material

Background reading, architecture write-ups, and past project reports that outlive any single
`openspec` change or agent's own harness. If a document explains *what something is* or *what
happened*, rather than tracking in-progress work, it belongs here, not in a change folder that
will eventually get archived.

Organized by subject, not by date or author — each subfolder is self-contained (any relative
links between files inside one subfolder keep working; nothing links across subfolders).

| Folder | Contents |
| --- | --- |
| [`agentbot/`](agentbot/) | AgentBot's architecture, system map, and runtime topology diagrams |
| [`eval-reports/`](eval-reports/) | Cross-model eval project reports (GR00T N1.6 vs N1.7, JTC interpolation) |
| [`pegasus/`](pegasus/) | The Pegasus H100 box's own automation report |
| [`pipeline/`](pipeline/) | The (pre-restructure) training pipeline's architecture |
| [`rldx1/`](rldx1/) | RLDX-1 background reading: an English field report and a Traditional Chinese close-reading of the paper, plus the figures both draw on. Moved here from `openspec/changes/add-rldx1-cansorting-eval/reports/` — that change tracks *evaluating* RLDX-1 and will eventually be archived; these two documents are reference material about the model itself and should stay findable after that happens. The *results* of that evaluation (the Phase-1 3-way comparison) stay in that change's own `tasks.md` — that's the change's actual output, not background reading. |
| [`setup-guides/`](setup-guides/) | Environment setup and CLI runbooks (IsaacLab/GR00T setup, 4090 delivery, CLI learning notes, control-rate runbook) |
| [`vla-hardware/`](vla-hardware/) | VLA-to-hardware control architecture and data-collector flow |

## Adding something here

Ask first: is this explaining something enduring (an architecture, a "what is X and how does it
work", a finished report), or is it tracking work still in progress? The latter belongs in an
`openspec` change or an agent's own `README.md`/`charter.md` instead.

Pick an existing folder if the new document is about the same subject; otherwise make a new one
— don't let this directory go back to being a flat, unsorted pile.
