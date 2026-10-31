"""
FantasyAI – expected FPL points for a player's next match.

  python3 predict_player.py "Salah"
  python3 predict_player.py "bruno fernandes"
  python3 predict_player.py              (asks for a name)

What it does
  1. Looks the player up in the live FPL API (accents and capitals don't matter).
  2. Finds his team's next match that hasn't kicked off.
  3. Builds the same pre-match features used in training (train_models.py),
     using your clean tables for past seasons and the live API for this season.
  4. Predicts with the TensorFlow network (main answer), plus gradient boosting
     and the last-5 average for comparison.

The first run trains both models on all of data/clean and saves them to models/
(about a minute). Later runs load them in seconds. Use --retrain after you
rebuild data/clean.

Needs train_models.py in the same folder.
Offline test: --local Fantasy-Premier-League uses that repo's snapshot of the
current season instead of the live API (its "next match" is as of the snapshot).
"""
import argparse
import json
import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_models as tm  # noqa: E402

API = "https://fantasy.premierleague.com/api/"
POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
STAT_COLS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_missed", "penalties_saved", "yellow_cards", "red_cards", "saves",
    "bonus", "bps", "influence", "creativity", "threat", "ict_index", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution",
    "value", "total_points",
]


# =========================================================================== #
# Data sources: live FPL API, or a local vaastav snapshot with the same shape  #
# =========================================================================== #
class LiveFPL:
    def __init__(self, base=API):
        import requests
        self.base = base.rstrip("/") + "/"
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "FantasyAI/1.0 (personal project)"
        self._boot = None

    def _get(self, path):
        import requests
        try:
            r = self.s.get(self.base + path, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.SSLError:
            sys.exit("Certificate error reaching the FPL API. On a Mac, run "
                     "'Install Certificates.command' in your Python folder, then try again.")
        except requests.exceptions.RequestException as e:
            sys.exit(f"Could not reach the FPL API ({e}). Check your connection, "
                     "or test offline with --local Fantasy-Premier-League.")

    def bootstrap(self):
        if self._boot is None:
            self._boot = self._get("bootstrap-static/")
        return self._boot

    def fixtures(self):
        return self._get("fixtures/")

    def history(self, element_id):
        return self._get(f"element-summary/{element_id}/")["history"]

    def season(self):
        start = int(self.bootstrap()["events"][0]["deadline_time"][:4])
        return f"{start}-{(start + 1) % 100:02d}"


class LocalSnapshot:
    """Reads a Fantasy-Premier-League clone; same structures as the live API."""

    def __init__(self, repo):
        data = Path(repo) / "data"
        seasons = sorted(p.parent.name for p in data.glob("*/fixtures.csv"))
        if not seasons:
            sys.exit(f"No season folders with fixtures.csv under {data}")
        self._season = seasons[-1]
        self.dir = data / self._season
        enc = dict(low_memory=False)
        self.players = pd.read_csv(self.dir / "players_raw.csv", **enc)
        self.teams = pd.read_csv(self.dir / "teams.csv", **enc)
        self.fx = pd.read_csv(self.dir / "fixtures.csv", **enc)
        self.gw = pd.read_csv(self.dir / "gws" / "merged_gw.csv", **enc)

    @staticmethod
    def _records(df):
        return json.loads(df.to_json(orient="records"))  # NaN -> None, like the API

    def bootstrap(self):
        return {"elements": self._records(self.players),
                "teams": self._records(self.teams),
                "events": []}

    def fixtures(self):
        return self._records(self.fx.drop(columns=["stats"], errors="ignore"))

    def history(self, element_id):
        h = self.gw[self.gw["element"] == element_id].rename(columns={"GW": "round_gw"})
        return self._records(h)

    def season(self):
        return self._season


# =========================================================================== #
# Finding the player                                                           #
# =========================================================================== #
# Letters that Unicode normalisation doesn't strip to plain a-z
SPECIAL = str.maketrans({"ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
                         "ß": "ss", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ð": "d",
                         "þ": "th", "ı": "i"})


def norm(text):
    text = unicodedata.normalize("NFKD", str(text).translate(SPECIAL))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def match_players(elements, query):
    """Players whose name matches the query: exact web/full name first, else all words."""
    q = norm(query)
    players = [e for e in elements if e.get("element_type") in POSITION]
    exact = [e for e in players
             if q in (norm(e["web_name"]), norm(f"{e['first_name']} {e['second_name']}"))]
    if exact:
        return exact
    tokens = q.split()
    return [e for e in players
            if all(t in norm(f"{e['first_name']} {e['second_name']} {e['web_name']}") for t in tokens)]


def find_player(elements, teams_by_id, query):
    found = match_players(elements, query)
    if not found:
        sys.exit(f"No player matching '{query}'. Try the surname as shown in FPL.")
    if len(found) == 1:
        return found[0]

    found.sort(key=lambda e: -float(e.get("total_points") or 0))
    print(f"\n{len(found)} players match '{query}':")
    for i, e in enumerate(found[:15], 1):
        print(f"  {i:>2}. {e['first_name']} {e['second_name']} ({e['web_name']}) – "
              f"{teams_by_id[e['team']]['short_name']}, {POSITION[e['element_type']]}, "
              f"£{e['now_cost'] / 10:.1f}m")
    if not sys.stdin.isatty():
        sys.exit("Be more specific, e.g. include the first name.")
    while True:
        pick = input("Choose a number: ").strip()
        if pick.isdigit() and 1 <= int(pick) <= min(15, len(found)):
            return found[int(pick) - 1]
        print("Please type one of the numbers above.")


# =========================================================================== #
# Building this season's rows in the same shape as data/clean                  #
# =========================================================================== #
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def is_done(f):
    return bool(f.get("finished") or f.get("finished_provisional"))


def season_team_rows(fixtures, season, code_of, target_ids):
    """Finished matches with scores, plus the upcoming fixtures in target_ids (no score)."""
    rows = []
    for f in fixtures:
        if not f.get("kickoff_time"):
            continue
        played = is_done(f)
        if not played and f["id"] not in target_ids:
            continue
        for side, opp in (("h", "a"), ("a", "h")):
            rows.append({
                "match_id": f"{season}_{int(f['id']):03d}", "season": season,
                "kickoff_time": f["kickoff_time"],
                "team_code": code_of[f[f"team_{side}"]], "opponent_code": code_of[f[f"team_{opp}"]],
                "is_home": side == "h",
                "goals_for": num(f.get(f"team_{side}_score")) if played else np.nan,
                "goals_against": num(f.get(f"team_{opp}_score")) if played else np.nan,
            })
    return pd.DataFrame(rows)


def season_player_rows(history, season, player, fixtures_by_id, code_of):
    rows = []
    for h in history:
        f = fixtures_by_id.get(h["fixture"])
        if f is None or not h.get("kickoff_time"):
            continue
        home = bool(h["was_home"])
        row = {
            "match_id": f"{season}_{int(h['fixture']):03d}", "season": season,
            "gameweek": h.get("round", h.get("round_gw")), "kickoff_time": h["kickoff_time"],
            "player_code": player["code"],
            "team_code": code_of[f["team_h"] if home else f["team_a"]],
            "opponent_code": code_of[h["opponent_team"]],
            "was_home": home, "position": POSITION[player["element_type"]],
        }
        row.update({c: num(h.get(c)) for c in STAT_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


def next_fixture(fixtures, team_id):
    upcoming = [f for f in fixtures
                if team_id in (f["team_h"], f["team_a"]) and f.get("kickoff_time")
                and not f.get("started") and not is_done(f)]
    if not upcoming:
        return None
    return min(upcoming, key=lambda f: f["kickoff_time"])


# =========================================================================== #
# Models: train once on all of data/clean, then reuse                          #
# =========================================================================== #
def data_stamp(data_dir):
    p = Path(data_dir) / "player_match_stats.csv"
    return f"{p.stat().st_size}-{int(p.stat().st_mtime)}"


def load_or_train(data_dir, model_dir, retrain=False):
    import joblib
    import tensorflow as tf

    model_dir = Path(model_dir)
    meta_p, tf_p, gbm_p = model_dir / "meta.json", model_dir / "tf_model.keras", model_dir / "gbm.joblib"
    stamp = data_stamp(data_dir)
    if not retrain and meta_p.exists() and tf_p.exists() and gbm_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta["data_stamp"] == stamp:
            tfb = (tf.keras.models.load_model(tf_p), pd.Series(meta["mu"]), pd.Series(meta["sd"]),
                   np.array(meta["class_values"], dtype="float32"), meta["epochs"])
            return tfb, joblib.load(gbm_p), meta["features"], meta["trained_to"]
        print("data/clean has changed since the models were trained – retraining.")

    print("Training models on all of data/clean (first run only, about a minute) ...")
    df, feats = tm.build_features(Path(data_dir))
    df = df.sort_values("kickoff_time").reset_index(drop=True)
    cut = int(len(df) * 0.92)            # latest 8% of matches used for early stopping
    train, val = df.iloc[:cut], df.iloc[cut:]
    gbm = tm.fit_gbm(train, val, feats)
    tfb = tm.fit_tf(train, val, feats)

    model_dir.mkdir(parents=True, exist_ok=True)
    tfb[0].save(tf_p)
    joblib.dump(gbm, gbm_p)
    trained_to = str(df["kickoff_time"].max().date())
    meta_p.write_text(json.dumps({
        "data_stamp": stamp, "features": feats, "mu": tfb[1].to_dict(), "sd": tfb[2].to_dict(),
        "class_values": tfb[3].tolist(), "epochs": tfb[4], "trained_to": trained_to,
        "rows": len(df)}, indent=1))
    print(f"Saved models to {model_dir.resolve()}")
    return tfb, gbm, feats, trained_to


# =========================================================================== #
# Output                                                                       #
# =========================================================================== #
def local_time(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()


def bar(p, width=30):
    return "█" * int(round(p * width))


def main():
    ap = argparse.ArgumentParser(description="Expected FPL points for a player's next match")
    ap.add_argument("name", nargs="*", help="player name, e.g. Salah or 'Bruno Fernandes'")
    ap.add_argument("--data", default="data/clean", help="folder with the five clean CSVs")
    ap.add_argument("--models", default="models", help="where trained models are saved")
    ap.add_argument("--retrain", action="store_true", help="retrain even if saved models exist")
    ap.add_argument("--local", metavar="REPO", help="use a Fantasy-Premier-League clone instead of the live API")
    ap.add_argument("--api", default=API, help=argparse.SUPPRESS)
    args = ap.parse_args()

    query = " ".join(args.name).strip() or input("Player name: ").strip()
    if not query:
        sys.exit("No name given.")

    src = LocalSnapshot(args.local) if args.local else LiveFPL(args.api)
    boot = src.bootstrap()
    teams_by_id = {t["id"]: t for t in boot["teams"]}
    code_of = {t["id"]: t["code"] for t in boot["teams"]}
    player = find_player(boot["elements"], teams_by_id, query)

    fixtures = src.fixtures()
    fixtures_by_id = {f["id"]: f for f in fixtures}
    fx = next_fixture(fixtures, player["team"])
    team = teams_by_id[player["team"]]
    pos = POSITION[player["element_type"]]
    name = f"{player['first_name']} {player['second_name']}"
    if fx is None:
        sys.exit(f"{name}: no upcoming {team['name']} match found in the FPL fixture list.")

    season = src.season()
    home = fx["team_h"] == player["team"]
    opp = teams_by_id[fx["team_a"] if home else fx["team_h"]]
    target_id = f"{season}_{int(fx['id']):03d}"

    # ---- assemble history: past seasons from data/clean, this season from FPL ----
    data_dir = Path(args.data)
    pms_all = pd.read_csv(data_dir / "player_match_stats.csv", low_memory=False)
    tms_all = pd.read_csv(data_dir / "team_match_stats.csv", low_memory=False)
    gw_of = pd.read_csv(data_dir / "matches.csv", usecols=["match_id", "gameweek"])

    past = pms_all[(pms_all["player_code"] == player["code"]) & (pms_all["season"] != season)]
    past = past.merge(gw_of, on="match_id", how="left")
    now = season_player_rows(src.history(player["id"]), season, player, fixtures_by_id, code_of)
    future = pd.DataFrame([{
        "match_id": target_id, "season": season, "gameweek": fx.get("event"),
        "kickoff_time": fx["kickoff_time"], "player_code": player["code"],
        "team_code": team["code"], "opponent_code": opp["code"], "was_home": home,
        "position": pos, "value": player["now_cost"],
    }])
    pms = pd.concat([past, now, future], ignore_index=True)
    pms["kickoff_time"] = pd.to_datetime(pms["kickoff_time"], utc=True, format="ISO8601")
    tms = pd.concat([tms_all[tms_all["season"] != season],
                     season_team_rows(fixtures, season, code_of, {fx["id"]})], ignore_index=True)
    tms["kickoff_time"] = pd.to_datetime(tms["kickoff_time"], utc=True, format="ISO8601")

    tfb, gbm, feats, trained_to = load_or_train(args.data, args.models, args.retrain)
    feats_df, _ = tm.make_features(pms, tms, keep_all=True)
    row = feats_df[feats_df["match_id"] == target_id]
    if row.empty:
        sys.exit("Could not build features for the upcoming match.")
    row = row.iloc[[0]]

    p_tf, pr_tf, P = tm.tf_predict(tfb, row.fillna(tfb[1]), feats)
    p_gbm, pr_gbm = tm.gbm_predict(gbm, row, feats)
    base = float(row["pts_mean5"].iloc[0])

    # ---- print ----
    when = local_time(fx["kickoff_time"])
    venue = "home" if home else "away"
    fixture_txt = f"{team['name']} v {opp['name']}" if home else f"{opp['name']} v {team['name']}"
    print(f"\n{name}  ({team['short_name']}, {pos}, £{player['now_cost'] / 10:.1f}m)")
    print(f"Next match: GW{fx.get('event')} – {fixture_txt} ({venue}), "
          f"{when:%a %d %b %Y, %H:%M} your time")

    chance = player.get("chance_of_playing_next_round")
    chance = None if chance is None else int(chance)
    flagged = player.get("status") not in (None, "a") or (chance is not None and chance < 100)
    if flagged:
        flag = f"{chance}% chance of playing" if chance is not None else f"status '{player.get('status')}'"
        print(f"⚠  FPL availability: {flag} – {player.get('news') or 'no details'}")
        print("   The model doesn't see injury news; its numbers assume he's as available as recently.")

    print("\nExpected points")
    print(f"  TensorFlow network     {p_tf[0]:5.1f}   ← main prediction")
    if flagged and chance is not None:
        print(f"    × FPL's {chance}% chance  {p_tf[0] * chance / 100:5.1f}   (rough adjustment for the injury flag)")
    print(f"  Gradient boosting      {p_gbm[0]:5.1f}")
    print(f"  Average of last 5      {base:5.1f}")

    print("\nChance of")
    for ev in tm.EVENTS:
        print(f"  {ev:<18} {pr_tf[ev][0]:5.0%}")

    probs = P[0]
    groups = [("0 or less", [0]), ("1", [1]), ("2", [2]), ("3", [3]), ("4–5", [4, 5]),
              ("6–9", list(range(6, 10))), ("10+", list(range(10, tm.N_CLASSES)))]
    print("\nPoints distribution (TensorFlow)")
    for label, idx in groups:
        p = probs[idx].sum()
        print(f"  {label:<10} {p:5.0%}  {bar(p)}")

    r = row.iloc[0]
    print("\nInputs that matter most")
    print(f"  Last 5 matches: {r['pts_mean5']:.1f} pts, {r['mins_mean5']:.0f} min, "
          f"{int(r['starts60_last5'])}/5 games of 60+ min")
    print(f"  Team goal diff per match (last 10): {r['team_gd10']:+.2f}   "
          f"Opponent: {r['opp_gd10']:+.2f}")
    if r["played_last5"] < 1 or r["matches_before"] < 5:
        print("  ⚠  Little recent playing history – treat this prediction with extra caution.")
    print(f"\nModels trained on matches up to {trained_to}. Typical miss: about ±1.6 points.")


if __name__ == "__main__":
    main()