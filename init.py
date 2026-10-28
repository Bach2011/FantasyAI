# past FPL data
import pandas as pd

seasons = ["2016-17","2017-18","2018-19","2019-20","2020-21",
           "2021-22","2022-23","2023-24","2024-25","2025-26"]
frames = []
for s in seasons:
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{s}/gws/merged_gw.csv"
    df = pd.read_csv(url, encoding="latin-1")  # older seasons are not UTF-8
    df["season"] = s
    frames.append(df)
fpl_gw = pd.concat(frames, ignore_index=True)
fpl_gw = fpl_gw.drop(columns=["xP"], errors="ignore")  # post-match leakage

# FPL Player Stats
from pathlib import Path
root = Path("FPL-Core-Insights/data/2025-2026/By Gameweek")
pms = pd.concat(
    [pd.read_csv(p).assign(gw=int(p.parent.name[2:]))
     for p in root.glob("GW*/playermatchstats.csv")],
    ignore_index=True)

# Football Results and Odds
import soccerdata as sd
mh = sd.MatchHistory(leagues="ENG-Premier League", seasons=list(range(2016, 2027)))
odds = mh.read_games()

#offical players FPL API
import truststore
truststore.inject_into_ssl()

import requests
boot = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
players = pd.DataFrame(boot["elements"])[["id","code","status","chance_of_playing_next_round","news","news_added"]]
players["captured_at"] = pd.Timestamp.utcnow()