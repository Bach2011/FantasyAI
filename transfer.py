"""
FantasyAI – recommend up to 3 FPL transfers for your squad.

  python3 transfer.py                         type your squad in when asked
  python3 transfer.py --squad my_squad.txt    read it from a file
  python3 transfer.py --squad my_squad.txt --horizon 5 --model gbm

Squad file: one player per line, (C) after your captain, then bank and free transfers.
Add the club after a comma if a name is ambiguous, and your selling price after
another comma if it differs from the current price (FPL shows it in the Transfers page).

    Raya
    Leno
    Gabriel
    Vuskovic, BHA
    Thomas, COV, 4.0
    ...
    Haaland (C)
    bank: 0.5
    free transfers: 1

For 0, 1, 2 and 3 transfers it finds the best possible squad (exactly, by integer
programming) that keeps FPL's rules:
  2 GK / 5 DEF / 5 MID / 3 FWD, at most 3 per club,
  money: bank + selling prices of players out >= prices of players in,
then subtracts 4 points for each transfer beyond your free transfers and recommends
the option with the best net gain. Gains are predicted points over --horizon
gameweeks (default 3): a transfer should pay off over several weeks, not one.
It also picks your best starting XI and captain for the next gameweek.

Needs team_builder.py, predict_player.py and train_models.py in the same folder.
"""
import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_player as pp  # noqa: E402
import team_builder as tb  # noqa: E402

HIT = 4          # points per extra transfer
MIN_GAIN = 1.0   # below this net gain over the horizon, saving the transfer is better


# =========================================================================== #
# 1. Reading the squad                                                        #
# =========================================================================== #
def parse_line(line):
    """'Haaland (C)', 'James, CHE', 'Thomas, COV, 4.0' -> (name, club, sell_price, captain)."""
    captain = bool(re.search(r"\(\s*c\s*\)|\*", line, re.I))
    line = re.sub(r"\(\s*c\s*\)|\*", "", line, flags=re.I)
    parts = [x.strip() for x in line.split(",") if x.strip()]
    name, club, sell = parts[0], None, None
    for x in parts[1:]:
        try:
            sell = float(x.replace("£", "").replace("m", ""))
        except ValueError:
            club = x.upper()
    return name, club, sell, captain


def resolve(elements, teams_by_id, name, club, interactive):
    found = pp.match_players(elements, name)
    if club:
        found = [e for e in found
                 if norm_club(teams_by_id[e["team"]]) == club or club in norm_club_names(teams_by_id[e["team"]])]
    if len(found) == 1:
        return found[0]
    if not found:
        msg = f"No player matching '{name}'" + (f" at {club}" if club else "")
        if interactive:
            print(f"  {msg}. Try again.")
            return None
        sys.exit(f"{msg}. Check the spelling against FPL.")
    if not interactive:
        options = "; ".join(f"{e['web_name']}, {teams_by_id[e['team']]['short_name']}" for e in found[:8])
        sys.exit(f"'{name}' matches several players ({options}). Add the club after a comma.")
    return pp.find_player(found, teams_by_id, name)


def norm_club(team):
    return team["short_name"].upper()


def norm_club_names(team):
    return pp.norm(team["name"]).upper()


def read_squad_file(path, elements, teams_by_id):
    picks, bank, ft = [], 0.0, 1
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("bank"):
            bank = float(re.sub(r"[^0-9.\-]", "", low.split(":", 1)[1]) or 0)
            continue
        if low.startswith("free"):
            ft = int(re.sub(r"[^0-9]", "", low.split(":", 1)[1]) or 1)
            continue
        name, club, sell, cap = parse_line(line)
        e = resolve(elements, teams_by_id, name, club, interactive=False)
        picks.append((e, sell, cap))
    return picks, bank, ft


def ask_squad(elements, teams_by_id):
    print("Enter your 15 players one at a time (add ', CLUB' if a name is ambiguous).")
    picks = []
    while len(picks) < 15:
        raw = input(f"  Player {len(picks) + 1}/15: ").strip()
        if not raw:
            continue
        name, club, sell, cap = parse_line(raw)
        e = resolve(elements, teams_by_id, name, club, interactive=True)
        if e is None:
            continue
        if any(p[0]["id"] == e["id"] for p in picks):
            print("  Already in your squad.")
            continue
        print(f"    ✓ {e['first_name']} {e['second_name']} ({teams_by_id[e['team']]['short_name']})")
        picks.append((e, sell, cap))
    if not any(p[2] for p in picks):
        while True:
            c = pp.norm(input("Who is your captain? ").strip())
            hit = [p for p in picks if c and c in pp.norm(f"{p[0]['first_name']} {p[0]['second_name']} {p[0]['web_name']}")]
            if len(hit) == 1:
                picks = [(e, s, e["id"] == hit[0][0]["id"]) for e, s, _ in picks]
                break
            print("  Type a name from your squad.")
    bank = float(input("Money in the bank (£m, e.g. 0.5): ").strip().replace("£", "") or 0)
    ft = int(input("Free transfers available (1-5): ").strip() or 1)
    if input("Save this squad to my_squad.txt for next time? [y/N] ").strip().lower() == "y":
        lines = [f"{e['web_name']}, {teams_by_id[e['team']]['short_name']}"
                 + (f", {s}" if s else "") + (" (C)" if c else "") for e, s, c in picks]
        Path("my_squad.txt").write_text("\n".join(lines + [f"bank: {bank}", f"free transfers: {ft}"]) + "\n",
                                        encoding="utf-8")
        print("  Saved. Next time run: python3 transfer.py --squad my_squad.txt")
    return picks, bank, ft


def check_squad(picks, teams_by_id):
    errs = []
    if len(picks) != 15:
        errs.append(f"{len(picks)} players entered; an FPL squad has 15")
    ids = [e["id"] for e, _, _ in picks]
    if len(set(ids)) != len(ids):
        errs.append("the same player is listed twice")
    pos = pd.Series([pp.POSITION[e["element_type"]] for e, _, _ in picks]).value_counts().to_dict()
    for p, n in tb.SQUAD.items():
        if pos.get(p, 0) != n:
            errs.append(f"{pos.get(p, 0)} {p} (need {n})")
    clubs = pd.Series([teams_by_id[e["team"]]["short_name"] for e, _, _ in picks]).value_counts()
    for club, n in clubs[clubs > tb.MAX_PER_CLUB].items():
        errs.append(f"{n} players from {club} (max {tb.MAX_PER_CLUB})")
    caps = sum(c for _, _, c in picks)
    if caps != 1:
        errs.append(f"{caps} captains marked (need exactly one '(C)')")
    if errs:
        sys.exit("Your squad doesn't look like a valid FPL squad: " + "; ".join(errs))


# =========================================================================== #
# 2. Finding the best transfers                                               #
# =========================================================================== #
def xi_points(table, xi, cap, col):
    return table.loc[xi, col].sum() + table.loc[cap, col]


def best_lineup(table, squad, col):
    """Best XI and captain from a fixed 15 for one points column."""
    t = table.loc[squad]
    _, xi, cap = tb.pick_squad(t, budget_tenths=10**6, bench_weight=0.0, points_col=col)
    return xi, cap


def objective(table, squad, col, bench_weight):
    xi, cap = best_lineup(table, squad, col)
    bench = [i for i in squad if i not in set(xi)]
    return xi_points(table, xi, cap, col) + bench_weight * table.loc[bench, col].sum()


def plan_transfers(table, owned_idx, sell, bank_tenths, ft, col, bench_weight, max_transfers,
                   pool_idx):
    t = table.loc[sorted(set(pool_idx) | set(owned_idx))].copy()
    t["owned"] = t.index.isin(owned_idx)
    t["cost"] = t["price"]
    for i, s in sell.items():
        t.loc[i, "cost"] = s
    budget = bank_tenths + int(t.loc[owned_idx, "cost"].sum())
    base = objective(t, list(owned_idx), col, bench_weight)

    options = []
    for k in range(0, max_transfers + 1):
        squad, _, _ = tb.pick_squad(t, budget, bench_weight, points_col=col, cost_col="cost",
                                    owned_col="owned", max_new=k)
        squad = list(squad)
        outs = [i for i in owned_idx if i not in set(squad)]
        ins = [i for i in squad if i not in set(owned_idx)]
        made = len(ins)
        gain = objective(t, squad, col, bench_weight) - base + 0.0
        gain = 0.0 if abs(gain) < 1e-9 else gain
        hit = HIT * max(0, made - ft)
        bank_after = budget - int(t.loc[squad, "cost"].sum())
        options.append({"k": k, "made": made, "squad": squad, "outs": outs, "ins": ins,
                        "gain": gain, "hit": hit, "net": gain - hit, "bank_after": bank_after})
    return options, base, t


def pair_up(t, outs, ins):
    """Show transfers like-for-like by position, most expensive first."""
    pairs = []
    for pos in tb.SQUAD:
        o = sorted([i for i in outs if t.loc[i, "position"] == pos], key=lambda i: -t.loc[i, "cost"])
        n = sorted([i for i in ins if t.loc[i, "position"] == pos], key=lambda i: -t.loc[i, "price"])
        pairs += list(zip(o, n))
    return pairs


# =========================================================================== #
# 3. Printing                                                                 #
# =========================================================================== #
def money(tenths):
    return f"£{tenths / 10:.1f}m"


def show(t, table, options, base, picks_cap_idx, ft, col, next_col, gws, model_label, horizon):
    gw_txt = f"GW{gws[0]}" if horizon == 1 else f"GW{gws[0]}–{gws[-1]}"
    print(f"\n=== {model_label}: predicted points over {gw_txt} ===")
    print(f"Your current squad: {base:.1f} pts (best XI + captain, bench at 10%)\n")
    print(f"  {'Option':<14}{'Gain':>7}{'Hit':>6}{'Net':>7}")
    best = max(options, key=lambda o: (round(o["net"], 6), -o["made"]))
    if best["made"] > 0 and best["net"] < MIN_GAIN:
        best = options[0]
    for o in options:
        label = "Keep the team" if o["k"] == 0 else f"{o['k']} transfer{'s' if o['k'] > 1 else ''}"
        if o["k"] > 0 and o["made"] < o["k"]:
            label += f" ({o['made']})"
        mark = "  ← recommended" if o is best else ""
        print(f"  {label:<14}{o['gain']:>+7.1f}{-o['hit']:>6}{o['net']:>+7.1f}{mark}")
    print(f"  (you have {ft} free transfer{'s' if ft != 1 else ''}; each extra one costs {HIT} pts)")

    for o in options[1:]:
        if not o["made"] or (options[o["k"] - 1]["made"] == o["made"] and o["k"] > 1):
            continue
        title = f"Best {o['made']} transfer{'s' if o['made'] > 1 else ''}"
        print(f"\n{title}  (gain {o['gain']:+.1f}, net {o['net']:+.1f}, "
              f"bank after {money(o['bank_after'])})")
        for out_i, in_i in pair_up(t, o["outs"], o["ins"]):
            r_o, r_i = t.loc[out_i], t.loc[in_i]
            print(f"  OUT {r_o['name']:<16}{r_o.team:<4} {money(r_o.cost):>7} {r_o[col]:5.1f} pts"
                  f"   →  IN {r_i['name']:<16}{r_i.team:<4} {money(r_i.price):>7} {r_i[col]:5.1f} pts"
                  f"   {r_i.fixtures}")

    if best["made"] == 0:
        top = options[1] if len(options) > 1 else None
        why = (f"the best transfer adds only {top['net']:+.1f} pts" if top and top["made"]
               else "no transfer improves your squad")
        print(f"\nRecommendation: save your transfer – {why}.")
    else:
        print(f"\nRecommendation: make the best {best['made']} transfer"
              f"{'s' if best['made'] > 1 else ''} above (net {best['net']:+.1f} pts over {gw_txt}).")

    # starting XI and captain for the next gameweek with the recommended squad
    xi, cap = best_lineup(t, best["squad"], next_col)
    bench = [i for i in best["squad"] if i not in set(xi)]
    gk_b = [i for i in bench if t.loc[i, "position"] == "GK"]
    out_b = sorted([i for i in bench if t.loc[i, "position"] != "GK"], key=lambda i: -t.loc[i, next_col])
    order = {p: n for n, p in enumerate(tb.SQUAD)}
    xi = sorted(xi, key=lambda i: (order[t.loc[i, "position"]], -t.loc[i, next_col]))
    form = "-".join(str(sum(t.loc[i, "position"] == p for i in xi)) for p in ("DEF", "MID", "FWD"))
    print(f"\nLine-up for GW{gws[0]} ({form}):")
    for pos in tb.SQUAD:
        names = [f"{t.loc[i, 'name']}{' (C)' if i == cap else ''} {t.loc[i, next_col]:.1f}"
                 for i in xi if t.loc[i, "position"] == pos]
        print(f"  {pos:<4}" + ", ".join(names))
    print("  Bench: " + ", ".join(f"{t.loc[i, 'name']} {t.loc[i, next_col]:.1f}" for i in gk_b + out_b))
    if picks_cap_idx in t.index and picks_cap_idx != cap:
        print(f"  Captain: {t.loc[cap, 'name']} ({t.loc[cap, next_col]:.1f}) rather than your "
              f"{t.loc[picks_cap_idx, 'name']} ({t.loc[picks_cap_idx, next_col]:.1f})")
    else:
        print(f"  Captain: keep {t.loc[cap, 'name']}")
    flagged = [i for i in best["squad"] if t.loc[i, "availability"] < 1]
    for i in flagged:
        print(f"  ⚠ {t.loc[i, 'name']}: {int(t.loc[i, 'availability'] * 100)}% chance of playing"
              f" – {t.loc[i, 'news'][:60]}")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Recommend up to 3 FPL transfers")
    ap.add_argument("--squad", help="squad file (see the top of this script); omit to type it in")
    ap.add_argument("--horizon", type=int, default=3, help="gameweeks to plan over (default 3)")
    ap.add_argument("--max-transfers", type=int, default=3)
    ap.add_argument("--model", choices=["both", "tf", "gbm", "mean5"], default="both")
    ap.add_argument("--bank", type=float, help="override the bank in the squad file (£m)")
    ap.add_argument("--free-transfers", type=int, help="override free transfers in the squad file")
    ap.add_argument("--bench-weight", type=float, default=0.1)
    ap.add_argument("--include-unproven", action="store_true",
                    help="allow buying players with no minutes in their last 5 matches")
    ap.add_argument("--data", default="data/clean")
    ap.add_argument("--models", default="models")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--local", metavar="REPO", help="use a Fantasy-Premier-League clone instead of the live API")
    ap.add_argument("--api", default=pp.API, help=argparse.SUPPRESS)
    args = ap.parse_args()

    src = pp.LocalSnapshot(args.local) if args.local else pp.LiveFPL(args.api)
    boot = src.bootstrap()
    teams_by_id = {t["id"]: t for t in boot["teams"]}
    elements = [e for e in boot["elements"] if e.get("element_type") in pp.POSITION]

    if args.squad:
        picks, bank, ft = read_squad_file(args.squad, elements, teams_by_id)
    else:
        picks, bank, ft = ask_squad(elements, teams_by_id)
    bank = args.bank if args.bank is not None else bank
    ft = args.free_transfers if args.free_transfers is not None else ft
    check_squad(picks, teams_by_id)

    table, gws, trained_to = tb.predict_all(src, args.data, args.models, args.horizon, args.retrain,
                                            must_include=[e["id"] for e, _, _ in picks])
    table = table.set_index("id", drop=False)
    owned = [e["id"] for e, _, _ in picks]
    sell = {e["id"]: int(round(s * 10)) for e, s, _ in picks if s is not None}
    cap_id = next(e["id"] for e, _, c in picks if c)
    pool = table.index if args.include_unproven else table.index[table["proven"]]

    chosen = ["tf", "gbm"] if args.model == "both" else [args.model]
    recs = {}
    for m in chosen:
        col, next_col = f"exp_{m}", f"next_{m}"
        options, base, t = plan_transfers(table, owned, sell, int(round(bank * 10)), ft, col,
                                          args.bench_weight, args.max_transfers, pool)
        show(t, table, options, base, cap_id, ft, col, next_col, gws, tb.MODELS[m], args.horizon)
        recs[m] = options

    if len(chosen) == 2:
        k = args.max_transfers
        a = {(tuple(sorted(o["outs"])), tuple(sorted(o["ins"]))) for o in recs["tf"][1:k + 1] if o["made"]}
        b = {(tuple(sorted(o["outs"])), tuple(sorted(o["ins"]))) for o in recs["gbm"][1:k + 1] if o["made"]}
        ins_a = {i for o in recs["tf"][1:] for i in o["ins"]}
        ins_b = {i for o in recs["gbm"][1:] for i in o["ins"]}
        both = sorted(ins_a & ins_b, key=lambda i: table.loc[i, "name"])
        print("\n=== Where the two models agree ===")
        print("  Players both models bring in (in any of their options): "
              + (", ".join(table.loc[i, "name"] for i in both) or "none"))
        if a & b:
            print("  Some transfer sets are identical in both models – the strongest signal.")
    print(f"\nPrices use FPL's current prices unless you gave selling prices. "
          f"Models trained on matches up to {trained_to}.")


if __name__ == "__main__":
    main()