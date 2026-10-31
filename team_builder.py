"""
FantasyAI – pick the FPL squad with the most predicted points.

  python3 team_builder.py                  next gameweek, £100.0m
  python3 team_builder.py --horizon 3      best squad over the next 3 gameweeks
  python3 team_builder.py --budget 99.5 --model gbm
  python3 team_builder.py --backtest results/test_predictions.csv

What it does
  1. Predicts every available player's points for each of his matches in the
     next --horizon gameweeks (double gameweeks count both matches), using the
     models saved by predict_player.py.
  2. Multiplies by FPL's chance of playing when a player is flagged.
  3. Solves for the best squad under FPL's rules, exactly (integer programming):
       15 players: 2 GK, 5 DEF, 5 MID, 3 FWD; at most 3 per club; cost <= budget
       starting XI: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD; captain scores double
     It maximises XI points + captain points + 10% of bench points (the bench
     only scores when a starter doesn't play).

--backtest replays past gameweeks with the out-of-sample predictions written by
train_models.py and reports how many points the chosen teams actually scored.

Needs predict_player.py and train_models.py in the same folder.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_player as pp  # noqa: E402
import train_models as tm  # noqa: E402

SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
MODELS = {"tf": "TensorFlow network", "gbm": "Gradient boosting", "mean5": "Average of last 5"}


# =========================================================================== #
# 1. The optimiser                                                            #
# =========================================================================== #
def pick_squad(players: pd.DataFrame, budget_tenths: int, bench_weight: float = 0.1,
               points_col: str = "exp_pts", cost_col: str = "price",
               owned_col: str | None = None, max_new: int | None = None):
    """players needs: position, team, cost_col (in £0.1m), and points_col.
    For transfers: owned_col marks the current squad, and at most max_new
    players from outside it may be brought in.
    Returns (squad, starters, captain) as index labels of `players`."""
    p = players.reset_index(drop=True)
    n = len(p)
    pts = p[points_col].to_numpy(float)
    # variables: [squad x (n) | starter s (n) | captain c (n)]
    obj = -np.concatenate([bench_weight * pts, (1 - bench_weight) * pts, pts])
    rows, lo, hi = [], [], []

    def add(coefs, a, b):
        rows.append(coefs); lo.append(a); hi.append(b)

    Z = np.zeros(n)
    I = np.eye(n)
    add(np.concatenate([np.ones(n), Z, Z]), 15, 15)
    add(np.concatenate([Z, np.ones(n), Z]), 11, 11)
    add(np.concatenate([Z, Z, np.ones(n)]), 1, 1)
    add(np.concatenate([p[cost_col].to_numpy(float), Z, Z]), 0, budget_tenths)
    if owned_col is not None and max_new is not None:
        new = (~p[owned_col].astype(bool)).to_numpy(float)
        add(np.concatenate([new, Z, Z]), 0, max_new)
    for pos in SQUAD:
        m = (p["position"] == pos).to_numpy(float)
        add(np.concatenate([m, Z, Z]), SQUAD[pos], SQUAD[pos])
        add(np.concatenate([Z, m, Z]), XI_MIN[pos], XI_MAX[pos])
    for team in p["team"].unique():
        m = (p["team"] == team).to_numpy(float)
        add(np.concatenate([m, Z, Z]), 0, MAX_PER_CLUB)
    A = np.vstack(rows)
    # starter only if in squad (s - x <= 0), captain only if starter (c - s <= 0)
    link = np.vstack([np.hstack([-I, I, 0 * I]), np.hstack([0 * I, -I, I])])
    A = np.vstack([A, link])
    lo += [-np.inf] * (2 * n)
    hi += [0] * (2 * n)

    res = milp(obj, constraints=LinearConstraint(A, lo, hi),
               integrality=np.ones(3 * n), bounds=(0, 1),
               options={"time_limit": 60})
    if res.x is None:
        raise RuntimeError(f"No valid squad found ({res.message}). Is the budget too low?")
    x = np.round(res.x).astype(int)
    idx = players.index.to_numpy()
    return idx[x[:n] == 1], idx[x[n:2 * n] == 1], idx[x[2 * n:] == 1][0]


# =========================================================================== #
# 2. Predicting every player for the upcoming gameweek(s)                     #
# =========================================================================== #
def fetch_histories(src, ids, cache_file: Path):
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    todo = [i for i in ids if str(i) not in cache]
    if todo:
        print(f"Downloading match history for {len(todo)} players "
              f"(saved to {cache_file.name}, so later runs today are instant) ...")

        live = not isinstance(src, pp.LocalSnapshot)

        def one(i):
            if live:
                time.sleep(0.05)  # be polite to the FPL servers
            return i, src.history(i)

        workers = 4 if live else 1
        with ThreadPoolExecutor(workers) as ex:
            for k, (i, h) in enumerate(ex.map(one, todo), 1):
                cache[str(i)] = h
                if k % 100 == 0:
                    print(f"  {k}/{len(todo)}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
    return {int(k): v for k, v in cache.items() if int(k) in set(ids)}


def upcoming_gameweeks(boot, fixtures, horizon):
    nxt = [e["id"] for e in boot.get("events", []) if e.get("is_next")]
    if nxt:
        first = nxt[0]
    else:  # local snapshot: earliest gameweek that still has a match to play
        open_gws = [f["event"] for f in fixtures
                    if f.get("event") and not f.get("started") and not pp.is_done(f)]
        if not open_gws:
            sys.exit("No upcoming gameweeks in the fixture list.")
        first = min(open_gws)
    return list(range(first, first + horizon))


def predict_all(src, data_dir, model_dir, horizon, retrain, must_include=()):
    """Predicted points for every available player over the next `horizon` gameweeks.
    Players in must_include (FPL ids) are always kept, even if unavailable or
    without a fixture (they then get 0 points)."""
    boot = src.bootstrap()
    teams_by_id = {t["id"]: t for t in boot["teams"]}
    code_of = {t["id"]: t["code"] for t in boot["teams"]}
    fixtures = src.fixtures()
    fixtures_by_id = {f["id"]: f for f in fixtures}
    season = src.season()
    gws = upcoming_gameweeks(boot, fixtures, horizon)
    targets = [f for f in fixtures if f.get("event") in gws and f.get("kickoff_time")
               and not f.get("started") and not pp.is_done(f)]
    if not targets:
        sys.exit(f"No unplayed fixtures in GW{gws[0]}–{gws[-1]}.")
    target_ids = {f["id"] for f in targets}

    elements = [e for e in boot["elements"]
                if e.get("element_type") in pp.POSITION
                and (e.get("status") != "u" or e["id"] in set(must_include))]
    stamp = f"{season}_gw{gws[0]}_{time.strftime('%Y%m%d')}"
    hist = fetch_histories(src, [e["id"] for e in elements],
                           Path(model_dir) / "cache" / f"histories_{stamp}.json")

    # past seasons from data/clean, this season from FPL, plus one row per upcoming match
    data_dir = Path(data_dir)
    pms_all = pd.read_csv(data_dir / "player_match_stats.csv", low_memory=False)
    gw_of = pd.read_csv(data_dir / "matches.csv", usecols=["match_id", "gameweek"])
    codes = {e["code"] for e in elements}
    past = pms_all[pms_all["player_code"].isin(codes) & (pms_all["season"] != season)]
    past = past.merge(gw_of, on="match_id", how="left")

    now_parts, fut_rows = [], []
    for e in elements:
        now_parts.append(pp.season_player_rows(hist.get(e["id"], []), season, e,
                                               fixtures_by_id, code_of))
        for f in targets:
            if e["team"] not in (f["team_h"], f["team_a"]):
                continue
            home = f["team_h"] == e["team"]
            fut_rows.append({
                "match_id": f"{season}_{int(f['id']):03d}", "season": season,
                "gameweek": f["event"], "kickoff_time": f["kickoff_time"],
                "player_code": e["code"], "team_code": code_of[e["team"]],
                "opponent_code": code_of[f["team_a"] if home else f["team_h"]],
                "was_home": home, "position": pp.POSITION[e["element_type"]],
                "value": e["now_cost"]})
    pms = pd.concat([past, *now_parts, pd.DataFrame(fut_rows)], ignore_index=True)
    pms["kickoff_time"] = pd.to_datetime(pms["kickoff_time"], utc=True, format="ISO8601")

    tms_all = pd.read_csv(data_dir / "team_match_stats.csv", low_memory=False)
    tms = pd.concat([tms_all[tms_all["season"] != season],
                     pp.season_team_rows(fixtures, season, code_of, target_ids)],
                    ignore_index=True)
    tms["kickoff_time"] = pd.to_datetime(tms["kickoff_time"], utc=True, format="ISO8601")

    tfb, gbm, feats, trained_to = pp.load_or_train(data_dir, model_dir, retrain)
    print("Predicting ...")
    feat, _ = tm.make_features(pms, tms, keep_all=True)
    fut_ids = {f"{season}_{int(i):03d}" for i in target_ids}
    rows = feat[feat["match_id"].isin(fut_ids)].copy()
    p_tf, _, _ = tm.tf_predict(tfb, rows.fillna(tfb[1]), feats)
    p_gbm, _ = tm.gbm_predict(gbm, rows, feats)
    rows["pred_tf"], rows["pred_gbm"], rows["pred_mean5"] = p_tf, p_gbm, rows["pts_mean5"]

    # opponent labels for display, e.g. "LIV (H), che (a)"
    short = {t["code"]: t["short_name"] for t in boot["teams"]}
    rows["opp_label"] = [f"{short[o]} (H)" if h else f"{short[o].lower()} (a)"
                         for o, h in zip(rows["opponent_code"], rows["home"] == 1)]
    rows = rows.sort_values("kickoff_time")
    first_row = rows.groupby("player_code").first()
    nxt = rows[rows["gameweek"] == gws[0]].groupby("player_code")[["pred_tf", "pred_gbm", "pred_mean5"]].sum()
    per_player = rows.groupby("player_code").agg(
        raw_tf=("pred_tf", "sum"), raw_gbm=("pred_gbm", "sum"), raw_mean5=("pred_mean5", "sum"),
        matches=("match_id", "size"),
        fixtures=("opp_label", ", ".join))

    info = pd.DataFrame([{
        "player_code": e["code"], "id": e["id"], "name": e["web_name"],
        "position": pp.POSITION[e["element_type"]],
        "team": teams_by_id[e["team"]]["short_name"], "price": int(e["now_cost"]),
        "status": e.get("status"), "chance": e.get("chance_of_playing_next_round"),
        "news": e.get("news") or ""} for e in elements]).set_index("player_code")
    out = info.join(per_player, how="left")
    out[["raw_tf", "raw_gbm", "raw_mean5", "matches"]] = \
        out[["raw_tf", "raw_gbm", "raw_mean5", "matches"]].fillna(0)   # no fixture = blank
    out["fixtures"] = out["fixtures"].fillna("no match")
    for m in MODELS:
        out[f"next_raw_{m}"] = nxt[f"pred_{m}"].reindex(out.index).fillna(0)
    out["played_last5"] = first_row["played_last5"]
    out["matches_before"] = first_row["matches_before"]
    out["pts_last5"] = first_row["pts_mean5"]

    chance = pd.to_numeric(out["chance"], errors="coerce")
    avail = np.where(chance.notna(), chance / 100.0, np.where(out["status"] == "a", 1.0, 0.5))
    out["availability"] = avail
    for m in MODELS:
        out[f"exp_{m}"] = out[f"raw_{m}"] * avail
        out[f"next_{m}"] = out[f"next_raw_{m}"] * avail     # first gameweek only
    # outside the population the model was trained on (no recent minutes)
    out["proven"] = (out["played_last5"] >= 1) & (out["matches_before"] >= 5) & (out["matches"] > 0)
    return out.reset_index(), gws, trained_to


# =========================================================================== #
# 3. Printing                                                                 #
# =========================================================================== #
def show_squad(table, squad, starters, captain, gws, budget, model_label):
    sq = table.loc[squad].copy()
    sq["start"] = sq.index.isin(starters)
    order = {p: i for i, p in enumerate(SQUAD)}
    xi = sq[sq["start"]].sort_values(["position", "exp_pts"], key=lambda s: s.map(order)
                                     if s.name == "position" else -s)
    bench = sq[~sq["start"]]
    bench = pd.concat([bench[bench.position == "GK"],
                       bench[bench.position != "GK"].sort_values("exp_pts", ascending=False)])
    form = "-".join(str((xi.position == p).sum()) for p in ("DEF", "MID", "FWD"))
    gw_txt = f"GW{gws[0]}" if len(gws) == 1 else f"GW{gws[0]}–{gws[-1]}"

    def line(r, tag=""):
        flag = f"  ⚠ {int(r.availability * 100)}% – {r.news[:40]}" if r.availability < 1 else ""
        price = f"£{r.price / 10:.1f}m"
        return (f"  {r.position:<4}{r['name'] + tag:<20}{r.team:<5}{price:>7}  "
                f"{r.exp_pts:5.1f}  {r.fixtures}{flag}")

    print(f"\nBest squad for {gw_txt}  ({model_label}, formation {form})")
    print(f"  {'':4}{'Player':<20}{'Club':<5}{'Price':>7}  {'Pts':>5}  Fixtures")
    for i, r in xi.iterrows():
        print(line(r, " (C)" if i == captain else ""))
    print("  Bench")
    for _, r in bench.iterrows():
        print(line(r))
    cost = sq["price"].sum()
    xi_pts = xi["exp_pts"].sum() + table.loc[captain, "exp_pts"]
    print(f"\n  Squad cost £{cost / 10:.1f}m of £{budget:.1f}m (£{(budget * 10 - cost) / 10:.1f}m left)")
    print(f"  Predicted points: {xi_pts:.1f} from the XI incl. captain, "
          f"bench {bench['exp_pts'].sum():.1f}")


def compare_squads(table, picks):
    (sq_a, xi_a, cap_a), (sq_b, xi_b, cap_b) = picks["tf"], picks["gbm"]
    common = sorted(set(sq_a) & set(sq_b), key=lambda i: table.loc[i, "name"])
    only_a = [table.loc[i, "name"] for i in sq_a if i not in set(sq_b)]
    only_b = [table.loc[i, "name"] for i in sq_b if i not in set(sq_a)]
    print("\nComparing the two squads")
    print(f"  In both squads ({len(common)}/15): " + ", ".join(table.loc[i, "name"] for i in common))
    print(f"  Only TensorFlow: {', '.join(only_a) or '–'}")
    print(f"  Only gradient boosting: {', '.join(only_b) or '–'}")
    print(f"  Captain: TensorFlow → {table.loc[cap_a, 'name']}, "
          f"gradient boosting → {table.loc[cap_b, 'name']}")
    # each starting XI scored by both models (captain doubled)
    print("\n  Predicted XI points       scored by TensorFlow   scored by gradient boosting")
    for label, xi, cap in (("TensorFlow squad", xi_a, cap_a), ("Gradient boosting squad", xi_b, cap_b)):
        vals = [table.loc[xi, f"exp_{m}"].sum() + table.loc[cap, f"exp_{m}"] for m in ("tf", "gbm")]
        print(f"  {label:<25}{vals[0]:>14.1f}{vals[1]:>24.1f}")
    print("  In backtests the gradient-boosting squad scored slightly more on average "
          "(61.7 v 58.7 pts per GW in 2025-26), but the gap wasn't statistically reliable.")


# =========================================================================== #
# 4. Backtest on past gameweeks (out-of-sample predictions)                   #
# =========================================================================== #
def backtest(pred_file, data_dir, budget_tenths, bench_weight, core_repo):
    preds = pd.read_csv(pred_file)
    pms = pd.read_csv(Path(data_dir) / "player_match_stats.csv",
                      usecols=["match_id", "player_code", "team_code", "value"])
    df = preds.merge(pms, on=["match_id", "player_code"], how="left")
    agg = (df.groupby(["fold", "season", "gameweek", "player_code"])
           .agg(position=("position", "first"), team=("team_code", "first"),
                price=("value", "first"), actual=("total_points", "sum"),
                tf=("pred_tensorflow", "sum"), gbm=("pred_gradient", "sum"),
                mean5=("pred_baseline", "sum"))
           .reset_index())

    avg_mgr = {}
    for season, folder in (("2025-26", "2025-2026"),):
        f = Path(core_repo) / "data" / folder / "gameweek_summaries.csv"
        if f.exists():
            g = pd.read_csv(f)
            avg_mgr.update({(season, int(r.id)): r.average_entry_score for r in g.itertuples()})

    models = {"TensorFlow network": "tf", "Gradient boosting": "gbm",
              "Average of last 5": "mean5", "Hindsight best (upper limit)": "actual"}
    rows = []
    for (fold, season, gw), g in agg.groupby(["fold", "season", "gameweek"]):
        g = g.reset_index(drop=True)
        rec = {"fold": fold, "season": season, "gameweek": gw}
        for label, col in models.items():
            squad, xi, cap = pick_squad(g, budget_tenths, bench_weight, points_col=col)
            rec[label] = g.loc[xi, "actual"].sum() + g.loc[cap, "actual"]
        rec["Average FPL manager"] = avg_mgr.get((season, int(gw)), np.nan)
        rows.append(rec)
    res = pd.DataFrame(rows)
    summary = res.groupby("fold")[list(models) + ["Average FPL manager"]].mean().round(1).T
    print("\nActual points per gameweek scored by each picked XI (captain doubled, no subs)")
    print(summary.to_string())
    for fold, r in res.groupby("fold"):
        beats = (r["TensorFlow network"] > r["Average of last 5"]).mean()
        print(f"  {fold}: TensorFlow team beat the last-5 team in {beats:.0%} of {len(r)} gameweeks")
    return res


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Best FPL squad from predicted points")
    ap.add_argument("--budget", type=float, default=100.0, help="in £m (default 100.0)")
    ap.add_argument("--horizon", type=int, default=1, help="number of gameweeks to plan for")
    ap.add_argument("--model", choices=["both", "tf", "gbm", "mean5"], default="both",
                    help="both = one squad from TensorFlow and one from gradient boosting")
    ap.add_argument("--bench-weight", type=float, default=0.1)
    ap.add_argument("--include-unproven", action="store_true",
                    help="allow players with no minutes in their last 5 matches")
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--models", default="models")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--local", metavar="REPO", help="use a Fantasy-Premier-League clone instead of the live API")
    ap.add_argument("--save", default="predictions_next.csv", help="CSV of every player's prediction")
    ap.add_argument("--backtest", metavar="TEST_PREDICTIONS_CSV")
    ap.add_argument("--core", default="FPL-Core-Insights",
                    help="FPL-Core-Insights clone, for the average-manager benchmark in --backtest")
    ap.add_argument("--api", default=pp.API, help=argparse.SUPPRESS)
    args = ap.parse_args()
    budget_tenths = int(round(args.budget * 10))

    if args.backtest:
        res = backtest(args.backtest, args.data, budget_tenths, args.bench_weight, args.core)
        res.to_csv("backtest_by_gameweek.csv", index=False)
        print("\nPer-gameweek results saved to backtest_by_gameweek.csv")
        return

    src = pp.LocalSnapshot(args.local) if args.local else pp.LiveFPL(args.api)
    table, gws, trained_to = predict_all(src, args.data, args.models, args.horizon, args.retrain)
    pool = table if args.include_unproven else table[table["proven"]]
    print(f"{len(pool)} players eligible ({len(table) - len(pool)} left out: "
          "no minutes in their last 5 matches or no match in this window; "
          "use --include-unproven to allow them)")

    chosen = ["tf", "gbm"] if args.model == "both" else [args.model]
    picks = {}
    for m in chosen:
        t = table.assign(exp_pts=table[f"exp_{m}"])
        squad, xi, cap = pick_squad(t.loc[pool.index], budget_tenths, args.bench_weight)
        picks[m] = (squad, xi, cap)
        show_squad(t, squad, xi, cap, gws, args.budget, MODELS[m])

    if len(chosen) == 2:
        compare_squads(table, picks)

    keep = ["name", "position", "team", "price", "fixtures", "exp_tf", "exp_gbm", "exp_mean5",
            "availability", "pts_last5", "proven", "news"]
    out = table.assign(_avg=(table["exp_tf"] + table["exp_gbm"]) / 2).sort_values("_avg", ascending=False)[keep]
    out.assign(price=out["price"] / 10).round(2).to_csv(args.save, index=False)
    print(f"\nEvery player's prediction saved to {args.save}. "
          f"Models trained on matches up to {trained_to}.")


if __name__ == "__main__":
    main()