What this is: a low-fit warning shown to the student before they pay to apply. Receipts/refunds/reconciliation (the team-wide focus this task) fix the money after the fact; this guardrail sits upstream of all of that — stop the doomed ₹100 payment from happening rather than only refunding it later.
File: p2_task08_spend_guardrail.py. Loads Task 7's tuned threshold if present (falls back to 0.5).
Three fit bands, not a single flag:

high_fit — meets threshold, no critical gaps, no warning
low_fit — below threshold but no critical gaps — borderline, warn and let them decide
very_low_fit — one or more critical gaps (>15pts under threshold) — strong warning, names the specific missing skills

Evaluation: treats "warned" as a prediction of "this application would have failed," so a guardrail that warns on everything (useless — blocks good matches too) or never warns (useless — protects nothing) both show up as a real number rather than looking fine in a demo.
Rupee-shaped metric: since this is a payments task, translated the guardrail's effect into ₹ — spend protected, spend still at risk, and friction cost (good matches wrongly warned) — not just abstract precision/recall.
Edge cases (3/3 passed): boundary p_match near the threshold, zero-skill student (must trigger very_low_fit, never silently high_fit), full warning coverage (every evaluated pair produces a valid fit_level — no gaps before payment).
Outputs: p2_task08_spend_guardrail.json (eval, spend-protection rupees, edge cases, demo), p2_task08_spend_guardrail.png (3-panel chart: warning quality, spend outcome breakdown, segment precision).
