"""
Build the five clean tables for the FantasyAI project from local clones of:
  - vaastav/Fantasy-Premier-League   (official FPL per-fixture data, 2016-17 onward)
  - olbauday/FPL-Core-Insights       (detailed match stats, 2025-26 onward)

Usage:
  python build_tables.py --vaastav Fantasy-Premier-League --core FPL-Core-Insights --out data/clean

Output (CSV, one file each):
  players.csv             one row per player ever        key: player_code
  teams.csv               one row per club ever          key: team_code
  matches.csv             one row per Premier League match  key: match_id
  player_match_stats.csv  one row per player per match   key: (match_id, player_code)
  team_match_stats.csv    one row per team per match     key: (match_id, team_code)

IDs used everywhere:
  player_code / team_code = FPL's stable codes (same value every season).
  FPL's `element` and team `id` are renumbered each season, so they are kept
  only as reference columns, never as join keys.
"""
import argparse
from pathlib import Path

import pandas as pd

POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# vaastav columns kept on player_match_stats (whatever exists in a given season)
FPL_STAT_COLS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_missed", "penalties_saved", "yellow_cards", "red_cards",
    "saves", "bonus", "bps", "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "clearances_blocks_interceptions", "recoveries",
    "tackles", "defensive_contribution", "value", "total_points",
]
# Deliberately dropped: xP (scraped after the gameweek -> leaks the result),
# selected / transfers_* (snapshot timing unclear), mng_* (2024-25 manager chip only).

# FPL-Core playermatchstats columns added for seasons it covers (prefix core_)
CORE_STAT_COLS = [
    "start_min", "finish_min", "total_shots", "shots_on_target", "xg", "xa", "xgot",
    "chances_created", "touches", "touches_opposition_box", "big_chances_missed",
    "successful_dribbles", "accurate_passes", "final_third_passes", "accurate_crosses",
    "tackles", "tackles_won", "interceptions", "recoveries", "blocks", "clearances",
    "duels_won", "duels_lost", "aerial_duels_won", "was_fouled", "fouls_committed",
    "offsides", "xgot_faced", "goals_prevented", "penalties_scored",
    "defensive_contributions",
]

# FPL-Core matches columns turned into per-team rows (home_X / away_X -> X)
CORE_TEAM_COLS = [
    "possession", "expected_goals_xg", "total_shots", "shots_on_target", "big_chances",
    "xg_open_play", "xg_set_play", "non_penalty_xg", "xg_on_target_xgot",
    "touches_in_opposition_box", "corners", "accurate_passes", "tackles_won",
    "interceptions", "blocks", "clearances", "keeper_saves", "fouls_committed",
]


def seasons_in(vaastav: Path):
    return sorted(p.parent.parent.name for p in (vaastav / "data").glob("*/gws/merged_gw.csv"))


def core_season_name(season: str) -> str:
    """vaastav '2025-26' -> FPL-Core '2025-2026'."""
    start = int(season[:4])
    return f"{start}-{start + 1}"


def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:  # 2016-17 to 2018-19 are latin-1
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


# --------------------------------------------------------------------------- #
# 1. Lookups from vaastav players_raw: element -> player_code, team id -> code
# --------------------------------------------------------------------------- #
def load_lookups(vaastav: Path, seasons):
    elem_rows, team_rows, player_rows = [], [], []
    for s in seasons:
        pr = read_csv_any(vaastav / "data" / s / "players_raw.csv")
        pr["season"] = s
        elem_rows.append(pr[["season", "id", "code", "element_type"]]
                         .rename(columns={"id": "element", "code": "player_code"}))
        team_rows.append(pr[["season", "team", "team_code"]].drop_duplicates()
                         .rename(columns={"team": "team_id"}))
        keep = [c for c in ["code", "first_name", "second_name", "web_name", "birth_date"] if c in pr]
        player_rows.append(pr[keep + ["season"]])
    return (pd.concat(elem_rows, ignore_index=True),
            pd.concat(team_rows, ignore_index=True),
            pd.concat(player_rows, ignore_index=True))


# --------------------------------------------------------------------------- #
# 2. players.csv and teams.csv
# --------------------------------------------------------------------------- #
def build_players(player_rows):
    # latest season's spelling wins; first season seen is kept for reference
    p = player_rows.sort_values("season")
    first = p.groupby("code")["season"].first().rename("first_season")
    last = p.groupby("code")["season"].last().rename("last_season")
    latest = p.drop_duplicates("code", keep="last").drop(columns="season")
    out = (latest.rename(columns={"code": "player_code"})
           .merge(first, left_on="player_code", right_index=True)
           .merge(last, left_on="player_code", right_index=True))
    return out.sort_values("player_code").reset_index(drop=True)


def build_teams(vaastav: Path, team_ids, seasons):
    names = []
    mtl = vaastav / "data" / "master_team_list.csv"
    if mtl.exists():
        names.append(pd.read_csv(mtl).rename(columns={"team": "team_id", "team_name": "name"}))
    for s in seasons:
        t = vaastav / "data" / s / "teams.csv"
        if t.exists():
            tt = pd.read_csv(t)[["id", "name", "short_name"]].rename(columns={"id": "team_id"})
            tt["season"] = s
            names.append(tt)
    names = pd.concat(names, ignore_index=True)
    t = team_ids.merge(names, on=["season", "team_id"], how="left").sort_values("season")
    seasons_played = t.groupby("team_code")["season"].apply(lambda x: ";".join(sorted(x.unique())))
    latest = t.dropna(subset=["name"]).drop_duplicates("team_code", keep="last")
    out = latest[["team_code", "name", "short_name"]].merge(
        seasons_played.rename("seasons"), left_on="team_code", right_index=True)
    # Fill these by hand later so ClubElo and Football-Data.co.uk can join on them
    out["clubelo_name"] = pd.NA
    out["footballdata_name"] = pd.NA
    return out.sort_values("team_code").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Per-season player rows from vaastav, with stable codes attached
# --------------------------------------------------------------------------- #
def load_vaastav_season(vaastav: Path, s, elem_map, team_map):
    gw = read_csv_any(vaastav / "data" / s / "gws" / "merged_gw.csv")
    gw["season"] = s
    # Known quirks: 2019-20 lists the COVID-postponed fixture twice (a 0-minute row
    # at the original GW29 date and the real one in "GW39"); 2025-26 has a few exact
    # duplicate rows. Keep each player's last row per fixture = the one actually played.
    gw = gw.sort_values("GW").drop_duplicates(["element", "fixture"], keep="last")
    gw = gw.merge(elem_map[elem_map.season == s].drop(columns="season"), on="element", how="left")
    tm = team_map[team_map.season == s].set_index("team_id")["team_code"]
    gw["opponent_code"] = gw["opponent_team"].map(tm)

    # Older seasons have no `team` column: derive each fixture's home/away teams.
    # Home-side players face the away team, and vice versa.
    home = (gw[~gw.was_home].groupby("fixture")["opponent_code"].first().rename("home_code"))
    away = (gw[gw.was_home].groupby("fixture")["opponent_code"].first().rename("away_code"))
    gw = gw.join(home, on="fixture").join(away, on="fixture")
    gw["team_code"] = gw["home_code"].where(gw.was_home, gw["away_code"])
    return gw


# --------------------------------------------------------------------------- #
# 4. matches.csv
# --------------------------------------------------------------------------- #
def build_matches(gw_all):
    m = (gw_all.sort_values("minutes", ascending=False)
         .drop_duplicates(["season", "fixture"])
         [["season", "fixture", "GW", "kickoff_time", "home_code", "away_code",
           "team_h_score", "team_a_score"]]
         .rename(columns={"GW": "gameweek", "home_code": "home_team_code",
                          "away_code": "away_team_code", "team_h_score": "home_goals",
                          "team_a_score": "away_goals"}))
    m["kickoff_time"] = pd.to_datetime(m["kickoff_time"], utc=True)
    m["match_id"] = m["season"] + "_" + m["fixture"].astype(int).astype(str).str.zfill(3)
    m["competition"] = "Premier League"
    cols = ["match_id", "season", "gameweek", "fixture", "kickoff_time", "competition",
            "home_team_code", "away_team_code", "home_goals", "away_goals"]
    return m[cols].sort_values("kickoff_time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. FPL-Core: player-match and match-level stats, keyed to our match_id
# --------------------------------------------------------------------------- #
def load_core(core: Path, s, matches):
    base = core / "data" / core_season_name(s) / "By Tournament" / "Premier League"
    if not base.exists():
        return None, None
    pms = pd.concat([pd.read_csv(f) for f in base.glob("GW*/playermatchstats.csv")], ignore_index=True)
    cm = pd.concat([pd.read_csv(f) for f in base.glob("GW*/matches.csv")], ignore_index=True)
    cm = cm[cm["finished"] == True].drop_duplicates("match_id")  # noqa: E712
    pms = pms.drop_duplicates(["player_id", "match_id"])

    # FPL-Core team ids are FPL team codes -> match on (home code, away code) within season
    cm = cm.rename(columns={"match_id": "core_match_id"})
    cm[["home_team", "away_team"]] = cm[["home_team", "away_team"]].astype(int)
    key = matches[matches.season == s][["match_id", "home_team_code", "away_team_code"]]
    cm = cm.merge(key, left_on=["home_team", "away_team"],
                  right_on=["home_team_code", "away_team_code"], how="inner")
    pms = pms.merge(cm[["core_match_id", "match_id"]], left_on="match_id",
                    right_on="core_match_id", how="inner", suffixes=("_core", ""))
    pms = pms.rename(columns={"player_id": "element"})
    keep = ["match_id", "element"] + [c for c in CORE_STAT_COLS if c in pms]
    pms = pms[keep].rename(columns={c: f"core_{c}" for c in CORE_STAT_COLS})
    return pms, cm


def core_team_rows(cm):
    rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        r = pd.DataFrame({"match_id": cm["match_id"], "team_code": cm[f"{side}_team"],
                          "elo_pre": cm[f"{side}_team_elo"], "opp_elo_pre": cm[f"{opp}_team_elo"]})
        for c in CORE_TEAM_COLS:
            if f"{side}_{c}" in cm:
                r[c] = cm[f"{side}_{c}"]
                if c in ("expected_goals_xg", "total_shots", "shots_on_target", "big_chances"):
                    r[f"{c}_against"] = cm[f"{opp}_{c}"]
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 6. team_match_stats.csv: goals for/against for every season, rich stats where available
# --------------------------------------------------------------------------- #
def build_team_match(matches, core_team):
    rows = []
    for side, opp in [("home", "away"), ("away", "home")]:
        rows.append(pd.DataFrame({
            "match_id": matches["match_id"], "season": matches["season"],
            "kickoff_time": matches["kickoff_time"],
            "team_code": matches[f"{side}_team_code"],
            "opponent_code": matches[f"{opp}_team_code"],
            "is_home": side == "home",
            "goals_for": matches[f"{side}_goals"], "goals_against": matches[f"{opp}_goals"],
        }))
    t = pd.concat(rows, ignore_index=True)
    if core_team is not None and len(core_team):
        t = t.merge(core_team, on=["match_id", "team_code"], how="left")
    return t.sort_values(["kickoff_time", "match_id", "is_home"],
                         ascending=[True, True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vaastav", required=True, type=Path)
    ap.add_argument("--core", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/clean"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seasons = seasons_in(args.vaastav)
    print("Seasons found:", ", ".join(seasons))
    elem_map, team_map, player_rows = load_lookups(args.vaastav, seasons)

    players = build_players(player_rows)
    teams = build_teams(args.vaastav, team_map, seasons)

    gw_all = pd.concat([load_vaastav_season(args.vaastav, s, elem_map, team_map) for s in seasons],
                       ignore_index=True)
    matches = build_matches(gw_all)

    # player_match_stats from vaastav
    gw_all = gw_all.merge(matches[["season", "fixture", "match_id"]], on=["season", "fixture"])
    gw_all["position"] = gw_all["element_type"].map(POSITION)
    pm_cols = (["match_id", "season", "player_code", "element", "team_code", "opponent_code",
                "was_home", "position"] + [c for c in FPL_STAT_COLS if c in gw_all])
    pms = gw_all[pm_cols].copy()

    # add FPL-Core detail where it exists
    core_players, core_teams = [], []
    for s in seasons:
        cp, cm = load_core(args.core, s, matches)
        if cp is None:
            continue
        cp["season"] = s
        core_players.append(cp)
        core_teams.append(core_team_rows(cm))
        print(f"  {s}: FPL-Core rows matched = {len(cp):,}")
    if core_players:
        pms = pms.merge(pd.concat(core_players), on=["season", "match_id", "element"], how="left")
    core_team = pd.concat(core_teams, ignore_index=True) if core_teams else None
    team_match = build_team_match(matches, core_team)

    pms = pms.merge(matches[["match_id", "kickoff_time"]], on="match_id")
    pms = pms.sort_values(["kickoff_time", "match_id", "team_code", "player_code"]).reset_index(drop=True)

    # ---- integrity checks: stop loudly instead of writing bad tables ----
    assert not pms.duplicated(["match_id", "player_code"]).any(), "duplicate player-match rows"
    assert pms["player_code"].notna().all(), "player rows without player_code"
    assert pms["team_code"].notna().all(), "player rows without team_code"
    assert matches["match_id"].is_unique
    assert set(pms["player_code"]) <= set(players["player_code"])
    assert set(teams["team_code"]) >= set(matches["home_team_code"]) | set(matches["away_team_code"])

    for name, df in [("players", players), ("teams", teams), ("matches", matches),
                     ("player_match_stats", pms), ("team_match_stats", team_match)]:
        df.to_csv(args.out / f"{name}.csv", index=False)
        print(f"{name + '.csv':26s} {len(df):>9,} rows  {df.shape[1]:>3} columns")


if __name__ == "__main__":
    main()