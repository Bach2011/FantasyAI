"""
FantasyAI – predict each player's FPL points for a match, and test the prediction.

Three models, all given only information available before kickoff:
  1. Baseline   – the player's mean points over his previous 5 matches.
  2. GBM        – gradient-boosted trees (scikit-learn) on the full feature set.
                  A strong standard benchmark: if the neural network can't beat
                  it, the network isn't worth its complexity.
  3. TensorFlow – a neural network that outputs a probability for every points
                  outcome (<=0, 1, 2, ... 14, 15+). Its expected value is the point
                  prediction; its probabilities are what calibration checks.

Testing is walk-forward (always train on the past, test on the future):
  Fold A: train 2016-17..2022-23, tune on 2023-24, test on 2024-25 (old rules)
  Fold B: train up to 2025-26 GW12, tune on GW13-19, test on GW20-38
          (2025-26 introduced defensive-contribution points)

Usage:
  python train_models.py --data data/clean --out results
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

SEED = 42
POSITIONS = ["GK", "DEF", "MID", "FWD"]
# Outcome classes for the neural network: <=0, 1, 2, ..., 14, 15+
CLASS_EDGES = list(range(1, 16))
N_CLASSES = len(CLASS_EDGES) + 1
EVENTS = {"return (4+ pts)": 4, "haul (10+ pts)": 10}


# =========================================================================== #
# 1. Features – everything is shifted so the current match is never included #
# =========================================================================== #
def prev_roll(g, col, n, how="mean"):
    s = g[col].shift(1)
    r = s.groupby(g.ngroup()).rolling(n, min_periods=1)
    out = r.mean() if how == "mean" else r.sum()
    return out.reset_index(level=0, drop=True)


def build_features(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    pms = pd.read_csv(data_dir / "player_match_stats.csv", low_memory=False)
    tms = pd.read_csv(data_dir / "team_match_stats.csv", low_memory=False)
    matches = pd.read_csv(data_dir / "matches.csv", usecols=["match_id", "gameweek"])

    pms["kickoff_time"] = pd.to_datetime(pms["kickoff_time"], utc=True)
    pms = pms.sort_values(["player_code", "kickoff_time", "match_id"]).reset_index(drop=True)
    pms = pms.merge(matches, on="match_id", how="left")
    pms["played"] = (pms["minutes"] > 0).astype(float)
    pms["started60"] = (pms["minutes"] >= 60).astype(float)
    pms["return4"] = (pms["total_points"] >= 4).astype(float)
    pms["haul10"] = (pms["total_points"] >= 10).astype(float)
    if "defensive_contribution" not in pms:
        pms["defensive_contribution"] = np.nan

    g = pms.groupby("player_code", sort=False)
    f = pd.DataFrame(index=pms.index)
    for n in (3, 5, 10):
        f[f"pts_mean{n}"] = prev_roll(g, "total_points", n)
    for n in (3, 5):
        f[f"mins_mean{n}"] = prev_roll(g, "minutes", n)
    f["mins_prev"] = g["minutes"].shift(1)
    f["starts60_last5"] = prev_roll(g, "started60", 5, "sum")
    f["played_last5"] = prev_roll(g, "played", 5, "sum")
    f["played_last10"] = prev_roll(g, "played", 10, "sum")
    for col, name in [("goals_scored", "goals"), ("assists", "assists"), ("bonus", "bonus"),
                      ("bps", "bps"), ("ict_index", "ict"), ("clean_sheets", "cs"),
                      ("saves", "saves"), ("goals_conceded", "gc"),
                      ("yellow_cards", "yellows"), ("expected_goals", "xg"),
                      ("expected_assists", "xa"), ("defensive_contribution", "defcon")]:
        f[f"{name}_mean5"] = prev_roll(g, col, 5)
    f["return_rate10"] = prev_roll(g, "return4", 10)   # share of last 10 with 4+ pts
    f["haul_rate10"] = prev_roll(g, "haul10", 10)      # share of last 10 with 10+ pts
    f["matches_before"] = g.cumcount()
    # season-to-date average; falls back to last-10 average at the start of a season
    gs = pms.groupby(["player_code", "season"], sort=False)["total_points"]
    f["pts_season_mean"] = gs.transform(lambda s: s.shift(1).expanding().mean())
    f["pts_season_mean"] = f["pts_season_mean"].fillna(f["pts_mean10"])
    f["price"] = pms["value"] / 10.0
    f["home"] = pms["was_home"].astype(float)

    # xG/xA exist from 2022-23, defensive contributions from 2025-26:
    # fill with 0 and add a flag so the model knows when the number is real.
    f["has_xg"] = f["xg_mean5"].notna().astype(float)
    f["has_defcon"] = f["defcon_mean5"].notna().astype(float)
    f[["xg_mean5", "xa_mean5", "defcon_mean5"]] = f[["xg_mean5", "xa_mean5", "defcon_mean5"]].fillna(0)
    f["new_rules"] = (pms["season"] >= "2025-26").astype(float)

    # Team and opponent strength from each side's previous 10 league matches
    tms["kickoff_time"] = pd.to_datetime(tms["kickoff_time"], utc=True)
    tms = tms.sort_values(["team_code", "kickoff_time"]).reset_index(drop=True)
    tms["gd"] = tms["goals_for"] - tms["goals_against"]
    tg = tms.groupby("team_code", sort=False)
    team = pd.DataFrame({"match_id": tms["match_id"], "team_code": tms["team_code"]})
    for col in ("goals_for", "goals_against", "gd"):
        team[f"{col}10"] = prev_roll(tg, col, 10)
    team["rest_days"] = (tms["kickoff_time"] - tg["kickoff_time"].shift(1)).dt.days.clip(upper=21)
    opp = team.drop(columns="rest_days").rename(
        columns={"team_code": "opponent_code", "goals_for10": "opp_goals_for10",
                 "goals_against10": "opp_goals_against10", "gd10": "opp_gd10"})
    team = team.rename(columns={"goals_for10": "team_goals_for10",
                                "goals_against10": "team_goals_against10", "gd10": "team_gd10"})

    base = pms[["match_id", "season", "gameweek", "kickoff_time", "player_code", "team_code",
                "opponent_code", "position", "total_points", "minutes"]]
    df = pd.concat([base, f], axis=1)
    df = df.merge(team, on=["match_id", "team_code"], how="left")
    df = df.merge(opp, on=["match_id", "opponent_code"], how="left")
    df["strength_diff"] = df["team_gd10"] - df["opp_gd10"]
    df["rest_days"] = df["rest_days"].fillna(7)
    for p in POSITIONS:
        df[f"pos_{p}"] = (df["position"] == p).astype(float)

    # Same population as the R analysis, decided before kickoff:
    # at least 5 previous listed matches and at least 1 appearance in the last 5.
    df = df[(df["matches_before"] >= 5) & (df["played_last5"] >= 1)].copy()
    df = df.dropna(subset=["team_gd10", "opp_gd10"]).reset_index(drop=True)

    feature_cols = [c for c in f.columns] + [
        "team_goals_for10", "team_goals_against10", "team_gd10", "opp_goals_for10",
        "opp_goals_against10", "opp_gd10", "strength_diff", "rest_days",
    ] + [f"pos_{p}" for p in POSITIONS]
    feature_cols = list(dict.fromkeys(feature_cols))
    df[feature_cols] = df[feature_cols].astype("float32")
    return df, feature_cols


# =========================================================================== #
# 2. Models                                                                   #
# =========================================================================== #
def to_class(points):
    return np.digitize(points, CLASS_EDGES)  # 0 = <=0 pts, 1 = 1 pt, ..., 15 = 15+ pts


def baseline_predict(df):
    """Mean of last 5 matches; probabilities = how often it happened in his last 10."""
    pts = df["pts_mean5"].to_numpy()
    probs = {
        "return (4+ pts)": df["return_rate10"].to_numpy(),
        "haul (10+ pts)": df["haul_rate10"].to_numpy(),
    }
    return pts, probs


def fit_gbm(train, val, feats):
    common = dict(max_iter=1000, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=None,
                  n_iter_no_change=30, random_state=SEED)
    # Early stopping watches the validation season (needs scikit-learn >= 1.4)
    reg = HistGradientBoostingRegressor(loss="squared_error", **common)
    reg.fit(train[feats], train["total_points"], X_val=val[feats], y_val=val["total_points"])
    clfs = {}
    for name, thr in EVENTS.items():
        clf = HistGradientBoostingClassifier(**common)
        clf.fit(train[feats], (train["total_points"] >= thr).astype(int),
                X_val=val[feats], y_val=(val["total_points"] >= thr).astype(int))
        clfs[name] = clf
    return reg, clfs


def gbm_predict(models, df, feats):
    reg, clfs = models
    return reg.predict(df[feats]), {k: c.predict_proba(df[feats])[:, 1] for k, c in clfs.items()}


def fit_tf(train, val, feats):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)

    mu = train[feats].mean()
    sd = train[feats].std().replace(0, 1)
    Xtr = ((train[feats] - mu) / sd).to_numpy("float32")
    Xva = ((val[feats] - mu) / sd).to_numpy("float32")
    ytr, yva = to_class(train["total_points"].to_numpy()), to_class(val["total_points"].to_numpy())

    # Points value represented by each class (mean of training points in that class)
    cls = to_class(train["total_points"].to_numpy())
    class_values = np.array([train["total_points"].to_numpy()[cls == k].mean() if (cls == k).any()
                             else float(k) for k in range(N_CLASSES)], dtype="float32")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(feats),)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy")
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                            restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=512,
                     callbacks=[stop], verbose=0)
    return model, mu, sd, class_values, len(hist.history["loss"])


def tf_predict(bundle, df, feats):
    model, mu, sd, class_values, _ = bundle
    X = ((df[feats] - mu) / sd).to_numpy("float32")
    P = model.predict(X, batch_size=4096, verbose=0)
    expected = P @ class_values
    probs = {name: P[:, to_class(np.array([thr]))[0]:].sum(axis=1) for name, thr in EVENTS.items()}
    return expected, probs, P


# =========================================================================== #
# 3. Metrics                                                                  #
# =========================================================================== #
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def calibration_table(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            rows.append({"bin": b, "n": int(m.sum()), "predicted": float(p[m].mean()),
                         "observed": float(y[m].mean())})
    t = pd.DataFrame(rows)
    ece = float(np.sum(t["n"] * (t["predicted"] - t["observed"]).abs()) / t["n"].sum())
    return t, ece


def bootstrap_diff(y, p_a, p_b, groups, metric, n=500, seed=SEED):
    """95% CI for metric(p_a) - metric(p_b), resampling whole matches."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(groups, return_inverse=True)
    rows_by = [np.flatnonzero(inv == i) for i in range(len(codes))]
    diffs = []
    for _ in range(n):
        pick = np.concatenate([rows_by[i] for i in rng.integers(0, len(codes), len(codes))])
        diffs.append(metric(y[pick], p_a[pick]) - metric(y[pick], p_b[pick]))
    return np.percentile(diffs, [2.5, 97.5]).tolist()


# =========================================================================== #
# 4. Run                                                                      #
# =========================================================================== #
FOLDS = {
    "A: test 2024-25": dict(
        train=lambda d: d["season"] <= "2022-23",
        val=lambda d: d["season"] == "2023-24",
        test=lambda d: d["season"] == "2024-25"),
    "B: test 2025-26 GW20-38": dict(
        train=lambda d: (d["season"] <= "2024-25") | ((d["season"] == "2025-26") & (d["gameweek"] <= 12)),
        val=lambda d: (d["season"] == "2025-26") & d["gameweek"].between(13, 19),
        test=lambda d: (d["season"] == "2025-26") & (d["gameweek"] >= 20)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean", type=Path)
    ap.add_argument("--out", default="results", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Building features ...")
    df, feats = build_features(args.data)
    print(f"  {len(df):,} player-matches, {len(feats)} features")

    metric_rows, calib_rows, preds_all = [], [], []
    for fold, spec in FOLDS.items():
        tr, va, te = df[spec["train"](df)], df[spec["val"](df)], df[spec["test"](df)]
        print(f"\n{fold}: train {len(tr):,} | validate {len(va):,} | test {len(te):,}")
        y = te["total_points"].to_numpy(dtype=float)

        p_base, pr_base = baseline_predict(te)
        print("  training gradient boosting ...")
        gbm = fit_gbm(tr, va, feats)
        p_gbm, pr_gbm = gbm_predict(gbm, te, feats)
        print("  training TensorFlow network ...")
        tfb = fit_tf(tr, va, feats)
        p_tf, pr_tf, _ = tf_predict(tfb, te, feats)
        print(f"    stopped after {tfb[4]} epochs")

        preds = {"Baseline (mean of last 5)": (p_base, pr_base),
                 "Gradient boosting": (p_gbm, pr_gbm),
                 "TensorFlow network": (p_tf, pr_tf)}
        out = te[["match_id", "season", "gameweek", "player_code", "position", "total_points"]].copy()
        out["fold"] = fold
        for name, (p, pr) in preds.items():
            short = name.split()[0].lower()
            out[f"pred_{short}"] = p
            r = {"fold": fold, "model": name, "n": len(y), "MAE": mae(y, p), "RMSE": rmse(y, p),
                 "bias (pred - actual)": float(np.mean(p - y))}
            for ev, thr in EVENTS.items():
                yy = (y >= thr).astype(float)
                pp = np.clip(pr[ev], 0, 1)
                t, ece = calibration_table(yy, pp)
                r[f"Brier {ev}"] = brier(yy, pp)
                r[f"ECE {ev}"] = ece
                t["fold"], t["model"], t["event"] = fold, name, ev
                calib_rows.append(t)
            # Practical FPL test: pick each gameweek's 10 highest predictions;
            # how many points did those players actually score on average?
            top = out.assign(_p=p).sort_values("_p", ascending=False).groupby("gameweek").head(10)
            r["top-10 picks per GW: mean actual pts"] = float(top["total_points"].mean())
            for pos in POSITIONS:
                m = (te["position"] == pos).to_numpy()
                r[f"MAE {pos}"] = mae(y[m], p[m])
            metric_rows.append(r)

        groups = te["match_id"].to_numpy()
        for other, p_other in [("Baseline (mean of last 5)", p_base), ("Gradient boosting", p_gbm)]:
            for mname, mf in [("MAE", mae), ("RMSE", rmse)]:
                lo, hi = bootstrap_diff(y, p_tf, p_other, groups, mf)
                metric_rows.append({"fold": fold, "model": f"TF minus {other}", "n": len(y),
                                    f"{mname} diff 95% CI": f"[{lo:.3f}, {hi:.3f}]"})
        preds_all.append(out)

    metrics = pd.DataFrame(metric_rows)
    calib = pd.concat(calib_rows, ignore_index=True)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    calib.to_csv(args.out / "calibration_bins.csv", index=False)
    pd.concat(preds_all).to_csv(args.out / "test_predictions.csv", index=False)
    make_charts(metrics, calib, pd.concat(preds_all), args.out)

    core = metrics[metrics["MAE"].notna()]
    cols = ["fold", "model", "MAE", "RMSE", "bias (pred - actual)", "top-10 picks per GW: mean actual pts",
            "Brier return (4+ pts)", "ECE return (4+ pts)", "ECE haul (10+ pts)"]
    print("\n" + core[cols].round(3).to_string(index=False))
    print("\n" + metrics[metrics["model"].str.startswith("TF minus")]
          .dropna(axis=1, how="all").to_string(index=False))
    print(f"\nSaved to {args.out.resolve()}")


def make_charts(metrics, calib, preds, out: Path):
    colors = {"Baseline (mean of last 5)": "#888780", "Gradient boosting": "#378ADD",
              "TensorFlow network": "#D85A30"}
    core = metrics[metrics["MAE"].notna()]
    folds = list(core["fold"].unique())

    # Chart 1: MAE and RMSE per model and fold
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, met in zip(axes, ["MAE", "RMSE"]):
        w = 0.26
        for i, (model, c) in enumerate(colors.items()):
            vals = [core[(core.fold == f) & (core.model == model)][met].iloc[0] for f in folds]
            bars = ax.bar(np.arange(len(folds)) + (i - 1) * w, vals, w, color=c, label=model)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(range(len(folds)), folds)
        ax.set_title(f"{met} (points, lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "1_mae_rmse.png", dpi=150)
    plt.close(fig)

    # Chart 2: calibration (reliability) curves for each event and fold
    events = list(EVENTS)
    fig, axes = plt.subplots(len(folds), len(events), figsize=(10, 4.5 * len(folds)), squeeze=False)
    for r, fold in enumerate(folds):
        for c, ev in enumerate(events):
            ax = axes[r, c]
            t = calib[(calib.fold == fold) & (calib.event == ev)]
            lim = max(t["predicted"].max(), t["observed"].max()) * 1.05
            ax.plot([0, lim], [0, lim], color="#B4B2A9", lw=1, ls="--")
            for model, col in colors.items():
                tt = t[t.model == model]
                ax.plot(tt["predicted"], tt["observed"], "o-", color=col, ms=4, label=model)
            ax.set_title(f"{fold}\nP({ev})", fontsize=10)
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "2_calibration.png", dpi=150)
    plt.close(fig)

    # Chart 3: calibration of the point prediction itself (deciles of prediction)
    fig, axes = plt.subplots(1, len(folds), figsize=(11, 4), squeeze=False)
    for i, fold in enumerate(folds):
        ax = axes[0, i]
        d = preds[preds.fold == fold]
        top = 0
        for model, col in colors.items():
            pc = "pred_" + model.split()[0].lower()
            q = pd.qcut(d[pc].rank(method="first"), 10, labels=False)
            g = d.groupby(q).agg(p=(pc, "mean"), a=("total_points", "mean"))
            ax.plot(g["p"], g["a"], "o-", color=col, ms=4, label=model)
            top = max(top, g["p"].max(), g["a"].max())
        ax.plot([0, top], [0, top], color="#B4B2A9", lw=1, ls="--")
        ax.set_title(f"{fold}: predicted vs actual points (10 groups)", fontsize=10)
        ax.set_xlabel("Mean predicted points")
        ax.set_ylabel("Mean actual points")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "3_points_calibration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()