import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE,
    ROOT,
    allowed,
    artifacts_ready,
    get_db,
    get_metrics,
    page_setup,
    sidebar_controls,
    sidebar_user,
    username,
)
from fraudkit.security import atomic_write_text
from fraudkit.score import ARTIFACTS

page_setup("Training Lab", "The latest training run, read from artifacts/. Retrain locally from here.")
LOG = ARTIFACTS / "training_log.jsonl"
STATUS = ARTIFACTS / "run_status.json"
STDOUT = ARTIFACTS / "retrain_stdout.log"
STAGES = ["load", "features", "split", "baseline", "tuning", "calibration", "threshold", "test", "save"]
if artifacts_ready():
    sidebar_controls()
else:
    sidebar_user()


def read_log() -> pd.DataFrame:
    if not LOG.exists():
        return pd.DataFrame()
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(json.JSONDecodeError):  # the last line may be half-written right now
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        tasklist = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "tasklist.exe"
        out = subprocess.run(
            [str(tasklist), "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True, check=False
        )
        return f'"{pid}"' in out.stdout  # exact match on the PID column
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_state() -> dict:
    if not STATUS.exists():
        return {"state": "idle"}
    s = json.loads(STATUS.read_text(encoding="utf-8"))
    if s.get("state") == "running" and not pid_alive(int(s.get("pid", -1))):
        s["state"] = "stopped"  # process died without writing a final status
    return s


# ----------------------------------------------------------------------------- retrain
state = run_state()
running = state.get("state") == "running"
with st.container(border=True):
    if not allowed("retrain"):
        st.caption("Retraining is limited to administrators. You can follow runs and results below.")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    fast = c1.toggle("Fast smoke run", value=True, help="Sample of cards, 3 trials, about 30 seconds")
    trials = c2.number_input("Optuna trials", 1, 200, 3 if fast else 30, disabled=fast)
    if c3.button("Retrain", type="primary", disabled=running or not allowed("retrain"), width="stretch"):
        cmd = [sys.executable, "-m", "fraudkit.train", "--config", str(ROOT / "configs" / "default.yaml"), "--no-live"]
        cmd += ["--fast"] if fast else ["--trials", str(int(trials))]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with open(STDOUT, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, creationflags=flags)
        # mark the run as active right away (the trainer rewrites this file with the same pid once it starts)
        atomic_write_text(STATUS, json.dumps({"state": "running", "pid": proc.pid, "started": "just now"}))
        get_db().log_admin(username(), "retrain_started", None, " ".join(cmd[2:]))
        st.session_state["retrain_started"] = True
        st.rerun()
    c4.markdown(
        f"<div class='fl-muted'>Status: <b>{state.get('state')}</b>"
        + (f" (pid {state.get('pid')}, started {state.get('started', '')})" if running else "")
        + "<br>Command: <code>python -m fraudkit.train --config configs/default.yaml</code></div>",
        unsafe_allow_html=True,
    )
    if running:
        st.caption(
            "Retrain is disabled while a run is active. Note: a new run overwrites the model and metrics "
            "shown on the other pages."
        )


@st.fragment(run_every=2 if running else None)
def live_tail():
    log = read_log()
    s = run_state()
    if s.get("state") == "running":
        st.info("Training in progress, this view refreshes every 2 seconds.")
    if log.empty:
        st.caption("No training log yet.")
        return
    starts = log[log["metric"] == "stage_start"]["stage"].tolist()
    ends = log[log["metric"] == "stage_end"]["stage"].tolist()
    current = next((x for x in starts if x not in ends), None)
    done = [x for x in STAGES if x in ends]
    st.progress(
        len(done) / len(STAGES),
        text=f"Stages done: {len(done)}/{len(STAGES)}" + (f" | running: {current}" if current else ""),
    )
    tail = log.tail(12)[["ts", "stage", "metric", "value"]].copy()
    tail["value"] = tail["value"].map(lambda v: f"{v:.5g}" if isinstance(v, float) else str(v)[:80])
    st.dataframe(tail.iloc[::-1], hide_index=True, width="stretch", height=250)
    if s.get("state") != "running" and st.session_state.get("retrain_started"):
        st.session_state["retrain_started"] = False
        st.session_state["retrain_done"] = s.get("state")
        st.cache_data.clear()
        st.rerun()  # full rerun: re-enable the button and reload charts from the new artifacts
    if st.session_state.get("retrain_done") == "finished":
        st.success("Run finished. Other pages now use the new model.")
    if s.get("state") == "failed":
        st.error(f"Last run failed: {s.get('error')}")


st.subheader("Run log")
live_tail()

if not artifacts_ready():
    st.stop()
m = get_metrics()
log = read_log()

# ----------------------------------------------------------------------------- timeline
st.subheader("Stage timeline")
if not log.empty and {"stage_start", "stage_end"} <= set(log["metric"]):
    t0 = pd.to_datetime(log["ts"].iloc[0])
    st_rows = []
    for s in STAGES:
        a = log[(log["stage"] == s) & (log["metric"] == "stage_start")]
        b = log[(log["stage"] == s) & (log["metric"] == "stage_end")]
        if len(a) and len(b):
            st_rows.append(
                {
                    "stage": s,
                    "start": pd.to_datetime(a["ts"].iloc[-1]),
                    "end": pd.to_datetime(b["ts"].iloc[-1]),
                    "seconds": float(b["value"].iloc[-1]),
                }
            )
    tl = pd.DataFrame(st_rows)
    if not tl.empty:
        fig = px.timeline(
            tl, x_start="start", x_end="end", y="stage", hover_data={"seconds": ":.1f"}, color_discrete_sequence=[BLUE]
        )
        fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=STAGES)
        fig.update_layout(height=320, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
c = st.columns(4)
c[0].metric("Total training time", f"{m['run']['total_seconds']:.0f} s")
c[1].metric("Optuna trials", m["tuning"]["trials"])
c[2].metric("Best CV PR-AUC", f"{m['tuning']['best_cv_pr_auc']:.4f}")
c[3].metric("Run type", "fast smoke run" if m["run"]["fast"] else "full run")

# ----------------------------------------------------------------------------- trials
st.subheader("Optuna trial history")
tr_path = ARTIFACTS / "trials.csv"
if tr_path.exists():
    tr = pd.read_csv(tr_path)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tr["trial"],
            y=tr["pr_auc_cv"],
            mode="markers",
            name="trial PR-AUC (CV)",
            marker=dict(color="#94A3B8", size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tr["trial"],
            y=tr["best_so_far"],
            mode="lines",
            line_shape="hv",
            name="best so far",
            line=dict(color=BLUE, width=3),
        )
    )
    fig.update_layout(height=340, xaxis_title="trial", yaxis_title="PR-AUC")
    st.plotly_chart(fig, width="stretch")
    with st.expander("All trials"):
        st.dataframe(tr, hide_index=True, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Final leaderboard")
    lb = pd.DataFrame(m["leaderboard"])[["model", "pr_auc", "roc_auc", "fit_seconds", "selected"]]
    st.dataframe(
        lb,
        hide_index=True,
        width="stretch",
        column_config={
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
        },
    )
    st.subheader("Chosen hyper-parameters")
    st.dataframe(
        pd.DataFrame(
            {
                "parameter": list(m["tuning"]["best_params"]),
                "value": [f"{v:.4g}" if isinstance(v, float) else str(v) for v in m["tuning"]["best_params"].values()],
            }
        ),
        hide_index=True,
        width="stretch",
    )
with right:
    st.subheader("Machine and libraries")
    mach = m["machine"]
    st.markdown(
        f"- **CPU threads used:** {mach['cpu_count']} (`n_jobs=-1`, LightGBM `num_threads={mach['cpu_count']}`)\n"
        f"- **Platform:** {mach['platform']}\n- **Processor:** {mach['processor']}"
    )
    st.dataframe(
        pd.DataFrame({"library": list(m["versions"]), "version": list(m["versions"].values())}),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Stage durations")
    du = pd.DataFrame({"stage": list(m["durations_s"]), "seconds": list(m["durations_s"].values())})
    st.dataframe(du, hide_index=True, width="stretch")
