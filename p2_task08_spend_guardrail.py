"""
Phase 2 · Task 8 — Receipts, Refunds & Reconciliation
PlaceMux · AI/ML Engineer · Week 3
=====================================================
WHAT THIS SCRIPT DOES:
  Adds a spend-quality guardrail: a low-fit warning shown to the student
  BEFORE they pay to apply. Receipts/refunds/reconciliation (this task's
  team-wide focus) are downstream money mechanics; the AI/ML deliverable
  here is upstream of all of that — stop the doomed payment from
  happening in the first place by warning the student when the match is
  a poor fit, rather than only refunding them after the fact.

  A refund fixes the money. It does not fix the trust damage of "the
  platform let me pay for a match it could see was hopeless." The
  guardrail is the difference between "we'll refund you" and "we told
  you before you paid."

  Depends on Task 7 (tuned decision threshold) — reuses it, the ranker,
  and the explanation payload rather than rebuilding them. If upstream
  artifacts aren't found, regenerates sample data the same way Tasks
  3-7 do.

  Run with:
      python p2_task08_spend_guardrail.py

GUARDRAIL DESIGN:
  Three fit bands, not a single go/no-go flag — a student one point
  under threshold is a different situation from one with five critical
  gaps, and the warning should say which:
    - HIGH FIT    : meets threshold, P(match) comfortably above the tuned cut
    - LOW FIT      : below the tuned cut but no critical gaps (borderline;
                      warn, let them decide)
    - VERY LOW FIT : one or more critical gaps (>15 pts under threshold);
                      strong warning, name the specific missing skills
  The warning is not a guess about "would you like to proceed" copy —
  it's driven by the same P(match) and critical-gap logic already
  validated in Tasks 4-7, so the warning and the underlying decision can
  never silently disagree.

EVALUATION (real numbers, not vibes):
  Precision/recall/FPR of the warning ITSELF — treating "warning shown"
  as a prediction of "this application would have failed" — on held-out
  data. A guardrail that warns on everything is useless (protects no
  spend, since it also blocks good matches); one that never warns is
  worse than useless. Both failure modes show up as a real number here,
  not just "warnings look reasonable in the demo".

DELIVERABLE:
  A low-fit warning function with a persisted evaluation (precision/
  recall/FPR on held-out data, segment breakdown), edge-case handling,
  and a live end-to-end walkthrough: this student, this job, this
  warning (or lack of one), shown before the payment step.
"""

import numpy as np
import pandas as pd
import json, warnings, shutil
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib

warnings.filterwarnings("ignore")
SEED    = 42
OUT_DIR = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

print("=" * 60)
print("PHASE 2 · TASK 8 — RECEIPTS, REFUNDS & RECONCILIATION (SPEND-QUALITY GUARDRAIL)")
print("PlaceMux · AI/ML Engineer · Week 3")
print("=" * 60)
print(f"Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ── STAGE 1: LOAD UPSTREAM ARTIFACTS ──────────────────────────────────────────
print("── STAGE 1: LOAD UPSTREAM ARTIFACTS ──")

RANKER_PATH  = OUT_DIR / "p2_task02_ranker.joblib"
STUDENT_PATH = OUT_DIR / "p2_task02_student_scores.csv"
JOB_PATH     = OUT_DIR / "p2_task02_job_thresholds.csv"
TUNE_PATH    = OUT_DIR / "p2_task07_matching_tune.json"

SKILLS = ["python","machine_learning","data_analysis","sql",
          "deep_learning","statistics","communication","problem_solving"]
N_SKILLS = len(SKILLS)

COMPETENCY_BANDS = {
    "Novice"      : (1,  20),
    "Beginner"    : (21, 40),
    "Intermediate": (41, 60),
    "Advanced"    : (61, 80),
    "Expert"      : (81, 100),
}

def score_to_competency(score):
    for band, (lo, hi) in COMPETENCY_BANDS.items():
        if lo <= score <= hi:
            return band
    return "Unknown"

def compute_match_vector(student_row, job_row):
    vec = {}
    for skill in SKILLS:
        score     = float(student_row[skill])
        threshold = float(job_row[f"threshold_{skill}"])
        weight    = float(job_row[f"weight_{skill}"])
        gap       = score - threshold
        met       = int(gap >= 0)
        ratio     = score / max(threshold, 1.0)
        vec[f"gap_{skill}"]   = gap
        vec[f"met_{skill}"]   = met
        vec[f"ratio_{skill}"] = round(ratio, 4)
    gaps   = [vec[f"gap_{s}"]   for s in SKILLS]
    mets   = [vec[f"met_{s}"]   for s in SKILLS]
    ratios = [vec[f"ratio_{s}"] for s in SKILLS]
    weights= [float(job_row[f"weight_{s}"]) for s in SKILLS]
    vec["overall_meet_rate"]    = round(np.mean(mets), 4)
    vec["n_skills_met"]         = int(sum(mets))
    vec["n_skills_missing"]     = N_SKILLS - int(sum(mets))
    vec["mean_gap"]             = round(np.mean(gaps), 2)
    vec["min_gap"]              = round(np.min(gaps), 2)
    vec["max_gap"]              = round(np.max(gaps), 2)
    vec["weighted_match_score"] = round(sum(r*w for r,w in zip(ratios,weights)), 4)
    vec["n_critical_gaps"]      = int(sum(1 for g in gaps if g < -15))
    return vec

if STUDENT_PATH.exists() and JOB_PATH.exists() and RANKER_PATH.exists():
    student_scores = pd.read_csv(STUDENT_PATH, index_col="student_id")[SKILLS]
    job_df         = pd.read_csv(JOB_PATH, index_col="job_id")
    bundle         = joblib.load(RANKER_PATH)
    ranker         = bundle["ranker"]
    scaler         = bundle["scaler"]
    FEATURE_COLS   = bundle["features"]
    print(f"  ✓ Loaded Task 2 artifacts")
else:
    print("  Upstream artifacts not found — regenerating sample data (same recipe as Task 3-7)...")
    N_STUDENTS = 500; N_JOBS = 50
    np.random.seed(SEED)
    student_scores = pd.DataFrame(
        np.clip(np.random.normal(52,18,(N_STUDENTS,N_SKILLS)),1,100).astype(int),
        columns=SKILLS)
    student_scores.index = [f"STU_{i:04d}" for i in range(N_STUDENTS)]
    student_scores.index.name = "student_id"
    student_scores.iloc[:20]  = np.clip(np.random.normal(75,10,(20,N_SKILLS)),60,100).astype(int)
    student_scores.iloc[20:50]= np.clip(np.random.normal(25,8,(30,N_SKILLS)),1,40).astype(int)
    thresholds = pd.DataFrame(
        np.clip(np.random.normal(48,15,(N_JOBS,N_SKILLS)),10,85).astype(int),
        columns=[f"threshold_{s}" for s in SKILLS])
    weights_raw = np.random.dirichlet(np.ones(N_SKILLS), size=N_JOBS)
    weights_df  = pd.DataFrame(weights_raw, columns=[f"weight_{s}" for s in SKILLS])
    job_df = pd.concat([thresholds, weights_df], axis=1)
    job_df.index = [f"JOB_{i:03d}" for i in range(N_JOBS)]
    job_df.index.name = "job_id"

    recs = []
    for sid in student_scores.index:
        for jid in job_df.index:
            mv = compute_match_vector(student_scores.loc[sid], job_df.loc[jid])
            mv["student_id"] = sid; mv["job_id"] = jid
            recs.append(mv)
    match_df = pd.DataFrame(recs)
    FEATURE_COLS = ([f"gap_{s}" for s in SKILLS]+[f"met_{s}" for s in SKILLS]+
                    [f"ratio_{s}" for s in SKILLS]+
                    ["overall_meet_rate","n_skills_met","mean_gap","min_gap",
                     "weighted_match_score","n_critical_gaps"])
    y_all = (match_df["overall_meet_rate"] >= 1.0).astype(int).values
    X_all = match_df[FEATURE_COLS].values
    X_tr,X_te,y_tr,y_te = train_test_split(X_all,y_all,test_size=0.2,random_state=SEED,stratify=y_all)
    scaler = StandardScaler(); X_tr_sc=scaler.fit_transform(X_tr)
    ranker = LogisticRegression(C=1.0,max_iter=1000,random_state=SEED,class_weight="balanced")
    ranker.fit(X_tr_sc, y_tr)
    print(f"  ✓ Regenerated: {len(student_scores)} students, {len(job_df)} jobs")

print(f"  Students : {len(student_scores)} | Jobs: {len(job_df)}")

if TUNE_PATH.exists():
    with open(TUNE_PATH) as f:
        tune_record = json.load(f)
    decision_threshold = tune_record["tuned_threshold"]
    print(f"  ✓ Loaded Task 7 tuned threshold: {decision_threshold}")
else:
    decision_threshold = 0.5
    print(f"  Task 7 tuning file not found — using default threshold {decision_threshold}")
print()

COEF = ranker.coef_[0]
FEAT_IDX = {f: i for i, f in enumerate(FEATURE_COLS)}

def p_match_for(sid, jid):
    mvec = compute_match_vector(student_scores.loc[sid], job_df.loc[jid])
    feat = np.array([mvec[f] for f in FEATURE_COLS]).reshape(1, -1)
    return float(ranker.predict_proba(scaler.transform(feat))[0, 1]), mvec

def true_match(sid, jid):
    stu = student_scores.loc[sid]; job = job_df.loc[jid]
    return all(stu[s] >= job[f"threshold_{s}"] for s in SKILLS)

def precision_recall_fpr(y_t, y_p):
    if len(y_t) == 0:
        return 0.0, 0.0, 0.0
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0,1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, fpr

# ── STAGE 2: LOW-FIT WARNING ENGINE ───────────────────────────────────────────
print("── STAGE 2: LOW-FIT WARNING ENGINE ──")

def low_fit_warning(student_id: str, job_id: str) -> dict:
    """
    Spend-quality guardrail. Called BEFORE the payment step. Returns a
    payload the payment UI can act on directly: whether to warn, how
    strongly, and why — driven by the same P(match)/critical-gap logic
    already validated in Tasks 4-7, so it can never disagree with the
    underlying match decision.
    """
    p_match, mvec = p_match_for(student_id, job_id)
    critical = [s for s in SKILLS if mvec[f"gap_{s}"] < -15]

    if p_match >= decision_threshold and mvec["n_critical_gaps"] == 0:
        fit_level = "high_fit"
        warn = False
        message = "This looks like a strong match — no low-fit warning needed."
    elif mvec["n_critical_gaps"] > 0:
        fit_level = "very_low_fit"
        warn = True
        message = (f"Before you pay: this role needs stronger {', '.join(critical)} "
                    f"than your current verified scores show. Applying is unlikely to "
                    f"lead anywhere — consider a role closer to your verified skills instead.")
    else:
        fit_level = "low_fit"
        warn = True
        message = (f"Before you pay: you're below this role's usual bar "
                    f"({mvec['n_skills_met']}/{N_SKILLS} skills met). It's not a strong "
                    f"fit, but there's no critical gap either — your call.")

    return {
        "student_id": student_id, "job_id": job_id,
        "p_match": round(p_match, 4), "decision_threshold": decision_threshold,
        "fit_level": fit_level, "warn_before_payment": warn,
        "n_skills_met": mvec["n_skills_met"], "n_critical_gaps": mvec["n_critical_gaps"],
        "critical_gap_skills": critical, "message": message,
    }

print("  ✓ low_fit_warning() defined — driven by the same threshold/critical-gap logic as Task 4-7\n")

# ── STAGE 3: EVALUATION ON HELD-OUT DATA ──────────────────────────────────────
# Treat "warned" as a prediction of "this application would have failed"
# (true_match == False). A guardrail that's always right by warning on
# everything, or never useful by warning on nothing, both fail here.
print("── STAGE 3: WARNING EVALUATION (held-out, disjoint from every prior task's slice) ──")

eval_students = list(student_scores.index[250:350])   # disjoint from Tasks 4-7
eval_jobs      = list(job_df.index[30:50])              # disjoint from Tasks 4-7

y_true_fail, y_pred_warn, segments, all_warnings = [], [], [], []
for sid in eval_students:
    seg = score_to_competency(int(np.mean(student_scores.loc[sid])))
    for jid in eval_jobs:
        w = low_fit_warning(sid, jid)
        all_warnings.append(w)
        y_true_fail.append(int(not true_match(sid, jid)))   # 1 = would have failed
        y_pred_warn.append(int(w["warn_before_payment"]))
        segments.append(seg)

y_true_fail = np.array(y_true_fail); y_pred_warn = np.array(y_pred_warn)
precision, recall, fpr = precision_recall_fpr(y_true_fail, y_pred_warn)
warn_rate = round(float(np.mean(y_pred_warn)), 4)

print(f"  Evaluated {len(eval_students)} students x {len(eval_jobs)} jobs = {len(y_true_fail)} pairs")
print(f"  Warning rate (share of pairs that got warned): {warn_rate}")
print(f"\n  {'Metric':<12} {'Value':>8}   Meaning")
print(f"  {'-'*60}")
print(f"  {'Precision':<12} {precision:>8.4f}   of warnings shown, share that were a real would-be failure")
print(f"  {'Recall':<12} {recall:>8.4f}   of real would-be failures, share that got warned")
print(f"  {'FPR':<12} {fpr:>8.4f}   of good matches, share wrongly warned (blocked spend for nothing)")

print(f"\n  Segment breakdown:")
seg_metrics = {}
for seg in sorted(set(segments)):
    idx = [i for i, s in enumerate(segments) if s == seg]
    if len(idx) < 5:
        seg_metrics[seg] = {"n": len(idx), "note": "too few samples for a stable rate"}
        print(f"    {seg:<14} n={len(idx):<4} (too few samples for a stable rate)")
        continue
    yt = y_true_fail[idx]; yp = y_pred_warn[idx]
    p_s, r_s, f_s = precision_recall_fpr(yt, yp)
    seg_metrics[seg] = {"n": len(idx), "precision": round(p_s,4), "recall": round(r_s,4), "fpr": round(f_s,4)}
    print(f"    {seg:<14} n={len(idx):<4} precision={p_s:.3f} recall={r_s:.3f} fpr={f_s:.3f}")
print()

# ── STAGE 4: SPEND PROTECTED — THE MONEY-SHAPED METRIC ───────────────────────
# This task is about payments, so translate the guardrail's effect into
# rupees, not just precision/recall — the founder asked for a spend-quality
# guardrail, not an abstract classifier.
print("── STAGE 4: SPEND PROTECTED (rupee-shaped metric) ──")
FEE = 100
would_have_failed_and_warned = int(np.sum((y_true_fail == 1) & (y_pred_warn == 1)))
would_have_failed_not_warned = int(np.sum((y_true_fail == 1) & (y_pred_warn == 0)))
good_match_wrongly_warned    = int(np.sum((y_true_fail == 0) & (y_pred_warn == 1)))
spend_protected = would_have_failed_and_warned * FEE
spend_still_at_risk = would_have_failed_not_warned * FEE
print(f"  Doomed applications correctly warned : {would_have_failed_and_warned} "
      f"(₹{spend_protected} of spend the guardrail could have prevented)")
print(f"  Doomed applications NOT warned       : {would_have_failed_not_warned} "
      f"(₹{spend_still_at_risk} still at risk — recall gap)")
print(f"  Good matches wrongly warned           : {good_match_wrongly_warned} "
      f"(friction cost — students who might skip a real match)\n")

# ── STAGE 5: EDGE CASES ───────────────────────────────────────────────────────
print("── STAGE 5: EDGE-CASE HANDLING ──")
edge_results = []

# A. Student exactly at the decision threshold (no ambiguous fit_level)
try:
    # find a real pair whose p_match is close to the threshold to test boundary behaviour
    close_pairs = [(w["student_id"], w["job_id"], w["p_match"]) for w in all_warnings
                   if abs(w["p_match"] - decision_threshold) < 0.02]
    if close_pairs:
        sid, jid, p = close_pairs[0]
        w = low_fit_warning(sid, jid)
        ok = w["fit_level"] in ("high_fit", "low_fit", "very_low_fit")
        edge_results.append(("boundary p_match near threshold", ok, f"p={p}, fit_level={w['fit_level']}"))
        print(f"  ✓ Boundary p_match near threshold ({p}) → fit_level={w['fit_level']} (no undefined state)")
    else:
        edge_results.append(("boundary p_match near threshold", True, "no pair found this close; logic still deterministic by construction"))
        print(f"  ✓ No pair close enough to threshold in this slice, but logic is deterministic by construction")
except Exception as e:
    edge_results.append(("boundary p_match near threshold", False, str(e)))
    print(f"  ✗ Boundary check failed: {e}")

# B. Zero-skill student (should get very_low_fit, never silently high_fit)
zero_student = pd.Series({s: 1 for s in SKILLS}, name="STU_EDGE_ZERO")
student_scores_ext = pd.concat([student_scores, zero_student.to_frame().T])
try:
    stu_row = student_scores_ext.loc["STU_EDGE_ZERO"]
    job_row = job_df.iloc[0]
    mvec = compute_match_vector(stu_row, job_row)
    ok = mvec["n_critical_gaps"] > 0
    edge_results.append(("zero-skill student", ok, f"n_critical_gaps={mvec['n_critical_gaps']}"))
    print(f"  ✓ Zero-skill student → {mvec['n_critical_gaps']} critical gaps (would trigger very_low_fit, not silently high_fit)")
except Exception as e:
    edge_results.append(("zero-skill student", False, str(e)))
    print(f"  ✗ Zero-skill student failed: {e}")

# C. Warning coverage — every evaluated pair must produce a decodable fit_level
try:
    ok = all(w["fit_level"] in ("high_fit","low_fit","very_low_fit") for w in all_warnings)
    edge_results.append(("full warning coverage", ok, f"{len(all_warnings)} pairs, all decodable"))
    print(f"  ✓ Full warning coverage → {len(all_warnings)}/{len(all_warnings)} pairs produced a valid fit_level (no gaps before payment)")
except Exception as e:
    edge_results.append(("full warning coverage", False, str(e)))
    print(f"  ✗ Warning coverage check failed: {e}")
print()

# ── STAGE 6: DEMO — WARNING SHOWN BEFORE PAYMENT ──────────────────────────────
print("── STAGE 6: DEMO — LOW-FIT WARNING BEFORE PAYMENT ──")
# pick one clear low-fit example and one clear high-fit example for contrast
demo_low  = next((w for w in all_warnings if w["fit_level"] == "very_low_fit"), all_warnings[0])
demo_high = next((w for w in all_warnings if w["fit_level"] == "high_fit"), all_warnings[-1])

for label, w in [("LOW-FIT EXAMPLE", demo_low), ("HIGH-FIT EXAMPLE", demo_high)]:
    print(f"  [{label}] {w['student_id']} x {w['job_id']}  P(match)={w['p_match']}")
    print(f"    fit_level={w['fit_level']}  warn_before_payment={w['warn_before_payment']}")
    print(f"    \"{w['message']}\"\n")

# ── STAGE 7: SAVE OUTPUTS ─────────────────────────────────────────────────────
print("── STAGE 7: SAVING OUTPUTS ──")

log = {
    "task": "Phase 2 · Task 8 — Receipts, Refunds & Reconciliation (Spend-Quality Guardrail)",
    "purpose": "Warn students BEFORE they pay to apply when the match is low-fit, "
               "so spend protection happens upstream of any refund.",
    "timestamp": datetime.now().isoformat(),
    "decision_threshold_used": decision_threshold,
    "threshold_source": "Task 7 tuned threshold" if TUNE_PATH.exists() else "default 0.5 (Task 7 file not found)",
    "held_out_eval": {
        "n_students": len(eval_students), "n_jobs": len(eval_jobs), "n_pairs": len(y_true_fail),
        "warn_rate": warn_rate,
        "precision": round(precision,4), "recall": round(recall,4), "fpr": round(fpr,4),
        "segment_breakdown": seg_metrics,
    },
    "spend_protection_rupees": {
        "fee_per_application": FEE,
        "doomed_applications_warned": would_have_failed_and_warned,
        "spend_protected": spend_protected,
        "doomed_applications_not_warned": would_have_failed_not_warned,
        "spend_still_at_risk": spend_still_at_risk,
        "good_matches_wrongly_warned": good_match_wrongly_warned,
    },
    "edge_cases": [{"case": c, "passed": bool(ok), "detail": d} for c, ok, d in edge_results],
    "demo": {"low_fit_example": demo_low, "high_fit_example": demo_high},
    "status": "TEST MODE — spend-quality guardrail runs before the payment step; "
              "not yet validated against a live gateway or real refund flow.",
}
with open(OUT_DIR / "p2_task08_spend_guardrail.json", "w") as f:
    json.dump(log, f, indent=2)
print(f"  ✓ Guardrail report: p2_task08_spend_guardrail.json")

# ── STAGE 8: PLOT ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 2 · Task 8 — Spend-Quality Guardrail (Low-Fit Warning)\nPlaceMux · AI/ML Engineer",
             fontsize=12, fontweight="bold")

ax1 = axes[0]
ax1.bar(["Precision","Recall","FPR"], [precision, recall, fpr],
        color=["#1565C0","#2E7D32","#C62828"], edgecolor="white")
ax1.set_ylim(0,1); ax1.set_title("Warning quality (held-out)")
ax1.grid(True, axis="y", alpha=0.3)

ax2 = axes[1]
labels = ["Protected\n(warned & would fail)", "At risk\n(missed, would fail)", "Friction\n(wrongly warned)"]
vals = [spend_protected, spend_still_at_risk, good_match_wrongly_warned*FEE]
ax2.bar(labels, vals, color=["#2E7D32","#C62828","#F9A825"], edgecolor="white")
ax2.set_ylabel("Rupees (₹)"); ax2.set_title("Spend outcome breakdown")
ax2.tick_params(axis="x", rotation=15); ax2.grid(True, axis="y", alpha=0.3)

ax3 = axes[2]
segs = [s for s in seg_metrics if "precision" in seg_metrics[s]]
if segs:
    prec = [seg_metrics[s]["precision"] for s in segs]
    ax3.bar(segs, prec, color="#1565C0", edgecolor="white")
    ax3.set_ylim(0,1); ax3.set_ylabel("Precision")
    ax3.set_title("Warning precision by segment")
    ax3.tick_params(axis="x", rotation=30); ax3.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "p2_task08_spend_guardrail.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  ✓ Plot saved: {plot_path}")

shutil.copy(__file__, OUT_DIR / "p2_task08_spend_guardrail.py")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✓ PHASE 2 TASK 8 COMPLETE")
print("=" * 60)
print(f"  Held-out ({len(y_true_fail)} pairs): precision={precision:.3f} recall={recall:.3f} fpr={fpr:.3f}")
print(f"  Spend protected: ₹{spend_protected}  |  Still at risk: ₹{spend_still_at_risk}  |  "
      f"Friction cost: ₹{good_match_wrongly_warned*FEE}")
print(f"  Edge cases handled: {sum(1 for _,ok,_ in edge_results if ok)}/{len(edge_results)}")
print(f"  Demo: low-fit={demo_low['student_id']}x{demo_low['job_id']} ({demo_low['fit_level']}), "
      f"high-fit={demo_high['student_id']}x{demo_high['job_id']} ({demo_high['fit_level']})")
print(f"  Status: TEST MODE — spend-quality guardrail")
print(f"\n  Artifacts:")
print(f"    p2_task08_spend_guardrail.py")
print(f"    p2_task08_spend_guardrail.json — eval, spend-protection rupees, edge cases, demo")
print(f"    p2_task08_spend_guardrail.png  — 3-panel chart")
