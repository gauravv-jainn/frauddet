# ===== Part E: PRO pipeline: honest temporal split, tuning, calibration, money-aware threshold, app export =====
import json, math, time as _time
import numpy as np, pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

VCOLS = [f"V{i}" for i in range(1, 29)]
FEATS = VCOLS + ["Amount", "hour_sin", "hour_cos"]
REVIEW_COST = 5.0                      # EUR an analyst spends reviewing one flagged transaction (editable in the app)
LOGIT_GRID = np.round(np.arange(-8, 8.0001, 0.1), 1)
P_GRID = 1 / (1 + np.exp(-LOGIT_GRID))

def make_features(d):
    """Raw Time grows over two days, so a model that memorises it fails on the future. Use time of day instead."""
    out = d[VCOLS + ["Amount"]].copy()
    ang = (d["Time"].values % 86400) / 86400 * 2 * np.pi
    out["hour_sin"], out["hour_cos"] = np.sin(ang), np.cos(ang)
    return out[FEATS]

def _sigmoid(z): return 1 / (1 + np.exp(-z))

def _export_trees(clf):
    td = clf.get_booster().trees_to_dataframe()
    trees = []
    for _, g in td.groupby("Tree", sort=True):
        g = g.reset_index(drop=True); idx = {nid: i for i, nid in enumerate(g["ID"])}; n = len(g)
        feat, thr, yes, no, miss, val, cov = [-1]*n, [0.0]*n, [-1]*n, [-1]*n, [-1]*n, [0.0]*n, [0.0]*n
        for i, r in g.iterrows():
            cov[i] = float(r["Cover"])
            if r["Feature"] == "Leaf": val[i] = float(r["Gain"])
            else:
                f = str(r["Feature"]); feat[i] = int(f[1:]) if f[1:].isdigit() else FEATS.index(f)
                thr[i] = float(r["Split"]); yes[i] = idx[r["Yes"]]; no[i] = idx[r["No"]]; miss[i] = idx[r["Missing"]]
        def fill(i):
            if feat[i] < 0: return val[i]
            a, b = fill(yes[i]), fill(no[i]); val[i] = (cov[yes[i]]*a + cov[no[i]]*b) / cov[i]; return val[i]
        fill(0)
        trees.append({"f": feat, "t": thr, "y": yes, "n": no, "m": miss, "v": val, "c": cov})
    return trees

def _margin(clf, X): return clf.get_booster().predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)), output_margin=True)

def _ops_table(y, p, amt):
    """for each probability threshold in the grid: tp, fp, fn, tn, caught EUR, missed EUR"""
    y = np.asarray(y).astype(int); rows = []
    for t in P_GRID:
        f = p >= t; pos = y == 1
        rows.append([int((f & pos).sum()), int((f & ~pos).sum()), int((~f & pos).sum()), int((~f & ~pos).sum()),
                     round(float(amt[f & pos].sum()), 2), round(float(amt[~f & pos].sum()), 2)])
    return rows

def _boot_ci(y, p, n_boot, seed):
    rs = np.random.default_rng(seed); y = np.asarray(y); vals = []
    for _ in range(n_boot):
        i = rs.integers(0, len(y), len(y))
        if y[i].sum() == 0: continue
        vals.append(average_precision_score(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def train_pro(df, seed=42, fast=False):
    t0 = _time.time()
    d = df.sort_values("Time").reset_index(drop=True)
    n = len(d); i1, i2 = int(n*0.6), int(n*0.8)
    tr, va, te = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    Xtr, Xva, Xte = [make_features(x).values.astype(np.float32) for x in (tr, va, te)]
    ytr, yva, yte = tr["Class"].values, va["Class"].values, te["Class"].values
    print(f"TEMPORAL split (past -> future): train {len(tr):,} ({ytr.sum()} fraud) | validation {len(va):,} ({yva.sum()}) | test {len(te):,} ({yte.sum()})")
    assert yva.sum() >= 5 and yte.sum() >= 5, "too few fraud cases in validation/test"
    ratio = (ytr == 0).sum() / max(ytr.sum(), 1)

    # ---- 1) leaderboard on VALIDATION (the test set is not touched until the very end)
    board = []
    def add(name, p): board.append({"model": name, "pr_auc": float(average_precision_score(yva, p)), "roc_auc": float(roc_auc_score(yva, p))})
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, ytr); add("Logistic Regression", lr.predict_proba(Xva)[:, 1])
    rf = RandomForestClassifier(n_estimators=60 if fast else 200, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed).fit(Xtr, ytr); add("Random Forest", rf.predict_proba(Xva)[:, 1])
    base = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=seed, scale_pos_weight=math.sqrt(ratio)).fit(Xtr, ytr); add("XGBoost (default)", base.predict_proba(Xva)[:, 1])

    # ---- 2) hyper-parameter search (stratified CV on the training window, metric = PR-AUC)
    space = {"max_depth": [3, 4, 5, 6], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [150, 250] if fast else [200, 350, 500],
             "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5], "reg_lambda": [1, 5, 10],
             "scale_pos_weight": [1.0, math.sqrt(ratio), ratio]}
    srch = RandomizedSearchCV(XGBClassifier(tree_method="hist", n_jobs=-1, random_state=seed), space, n_iter=3 if fast else 14,
                              scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=seed), random_state=seed, n_jobs=1, refit=True)
    srch.fit(Xtr, ytr); clf = srch.best_estimator_
    add("XGBoost (tuned)  <- used in app", clf.predict_proba(Xva)[:, 1])
    print("best params:", srch.best_params_)

    # ---- 3) calibration (Platt scaling on the margin) fitted on VALIDATION
    mva = _margin(clf, Xva)
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(mva.reshape(-1, 1), yva)
    a, b = float(pl.coef_[0, 0]), float(pl.intercept_[0])
    cal = lambda m: _sigmoid(a * m + b)
    pva = cal(mva)
    raw_va = _sigmoid(mva)
    print(f"calibration: Brier before {brier_score_loss(yva, raw_va):.5f} -> after {brier_score_loss(yva, pva):.5f}")

    # ---- 4) money-aware threshold chosen on VALIDATION: loss = missed fraud EUR + review cost per flag
    amt_va, amt_te = va["Amount"].values, te["Amount"].values
    def loss(tab, rc): return [r[5] + rc * (r[0] + r[1]) for r in tab]
    tab_va = _ops_table(yva, pva, amt_va)
    k = int(np.argmin(loss(tab_va, REVIEW_COST))); thr = float(P_GRID[k])
    print(f"validated threshold (min EUR loss, review cost {REVIEW_COST}): calibrated p >= {thr:.4f}")

    # ---- 5) FINAL test, untouched until now
    mte = _margin(clf, Xte); pte = cal(mte)
    tab_te = _ops_table(yte, pte, amt_te)
    ap = float(average_precision_score(yte, pte)); ci = _boot_ci(yte, pte, 40 if fast else 300, seed)
    kt = int(np.argmin(np.abs(P_GRID - thr))); tp, fp, fn, tn, caught, missed = tab_te[kt]
    base_loss = float(amt_te[yte == 1].sum())
    net = base_loss - (missed + REVIEW_COST * (tp + fp))
    print(f"TEST  PR-AUC {ap:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}) | ROC-AUC {roc_auc_score(yte, pte):.4f} | caught {tp}/{tp+fn} fraud | false alarms {fp} | net saving EUR {net:,.0f} of EUR {base_loss:,.0f} fraud exposure")

    pr_, rc_, _ = precision_recall_curve(yte, pte)
    sel = np.unique(np.linspace(0, len(pr_) - 1, min(150, len(pr_))).astype(int))
    rel_edges = [0, .001, .005, .01, .05, .1, .25, .5, .75, .9, 1.0000001]
    rel = []
    for lo, hi in zip(rel_edges[:-1], rel_edges[1:]):
        m = (pte >= lo) & (pte < hi)
        if m.sum(): rel.append([lo, min(hi, 1.0), int(m.sum()), int(yte[m].sum()), float(pte[m].mean())])
    lg = np.log(np.clip(pte, 1e-6, 1 - 1e-6) / (1 - np.clip(pte, 1e-6, 1 - 1e-6)))
    bins = np.arange(-12, 12.01, 1.0)
    hist_f, _ = np.histogram(lg[yte == 1], bins=bins); hist_g, _ = np.histogram(lg[yte == 0], bins=bins)
    sub = np.random.default_rng(seed).choice(len(Xte), min(15000, len(Xte)), replace=False)
    shap = clf.get_booster().predict(xgb.DMatrix(Xte[sub]), pred_contribs=True)[:, :-1]
    imp = np.abs(shap).mean(0)

    # ---- demo stream for the live monitor: test rows in time order, fraud boosted to ~5% so something happens
    rs = np.random.default_rng(seed)
    pos = np.where(yte == 1)[0]; neg = np.where(yte == 0)[0]
    npos = min(len(pos), 36); nneg = min(len(neg), int(npos / 0.05) - npos if npos else 600)
    pick = np.sort(np.concatenate([rs.choice(pos, npos, replace=False), rs.choice(neg, nneg, replace=False)]))
    stream = te.iloc[pick][["Time"] + VCOLS + ["Amount", "Class"]].round(5)

    P = {"clf": clf, "platt": [a, b], "thr": thr, "review_cost": REVIEW_COST, "seed": seed,
         "board": board, "params": {k_: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k_, v in srch.best_params_.items()},
         "split": {"train": [len(tr), int(ytr.sum())], "val": [len(va), int(yva.sum())], "test": [len(te), int(yte.sum())]},
         "test": {"pr_auc": ap, "ci": ci, "roc_auc": float(roc_auc_score(yte, pte)), "tab": tab_te, "base_loss": base_loss, "n": int(len(yte)), "pos": int(yte.sum()),
                  "pr_curve": [[float(rc_[i]), float(pr_[i])] for i in sel], "rel": rel,
                  "hist": {"bins": bins.tolist(), "fraud": hist_f.tolist(), "genuine": hist_g.tolist()}},
         "val_thr_idx": k, "imp": sorted(zip(FEATS, imp.round(4).tolist()), key=lambda z: -z[1])[:15],
         "brier": [float(brier_score_loss(yva, raw_va)), float(brier_score_loss(yva, pva))],
         "stream": stream, "Xte": Xte, "pte": pte, "mte": mte, "seconds": round(_time.time() - t0)}
    print(f"training pipeline done in {P['seconds']}s")
    return P

def build_app(P, template_path, out_path, data_source):
    clf = P["clf"]
    xt = P["Xte"][:1]; marg = float(_margin(clf, xt)[0]); trees = _export_trees(clf); s = 0.0
    for tr in trees:
        k = 0
        while tr["f"][k] >= 0:
            v = float(xt[0, tr["f"][k]]); k = tr["m"][k] if math.isnan(v) else (tr["y"][k] if v < tr["t"][k] else tr["n"][k])
        s += tr["v"][k]
    M = {"feats": FEATS, "vcols": VCOLS, "trees": trees, "base": marg - s, "platt": P["platt"], "thr": P["thr"], "rc": P["review_cost"],
         "grid": LOGIT_GRID.tolist(), "val_idx": P["val_thr_idx"], "board": P["board"], "params": P["params"], "split": P["split"],
         "test": P["test"], "imp": P["imp"], "brier": P["brier"], "source": data_source,
         "stream": {"cols": list(P["stream"].columns), "rows": P["stream"].values.tolist()}}
    html = open(template_path, encoding="utf-8").read().replace("__DATA_JSON__", json.dumps(M, separators=(",", ":")))
    open(out_path, "w", encoding="utf-8").write(html)
    return out_path
