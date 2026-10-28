# =============================================================================
# FantasyAI – How do pre-match factors relate to actual fantasy points?
#
# Input : the five CSVs made by build_tables.py (default folder: data/clean)
# Output: eda/  – PNG charts, one combined PDF, and a CSV of correlations
#
# Run from your FantasyAI folder:
#   Rscript eda_factors_vs_points.R data/clean eda
#
# Every factor is built ONLY from information available before kickoff
# (previous matches, price, home/away, opponent strength). Same-match stats
# such as this match's minutes, goals or xG are never used as factors, because
# they are part of the outcome, not a predictor of it.
# =============================================================================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr)
  library(ggplot2); library(forcats); library(scales); library(patchwork)
})

args    <- commandArgs(trailingOnly = TRUE)
in_dir  <- if (length(args) >= 1) args[1] else "data/clean"
out_dir <- if (length(args) >= 2) args[2] else "eda"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

pos_levels <- c("GK", "DEF", "MID", "FWD")
pos_cols   <- c(GK = "#7F77DD", DEF = "#1D9E75", MID = "#D85A30", FWD = "#378ADD")

# ---- helper: mean of the previous n values (excludes the current row) -------
roll_prev <- function(x, n) {
  x   <- lag(as.numeric(x))
  ok  <- !is.na(x)
  cs  <- cumsum(ifelse(ok, x, 0))
  cnt <- cumsum(ok)
  s   <- cs  - lag(cs,  n, default = 0)
  k   <- cnt - lag(cnt, n, default = 0)
  ifelse(k > 0, s / k, NA_real_)
}
roll_prev_sum <- function(x, n) {
  x  <- lag(as.numeric(x)); x[is.na(x)] <- 0
  cs <- cumsum(x)
  cs - lag(cs, n, default = 0)
}

# ---- 1. Load ----------------------------------------------------------------
pms <- read_csv(file.path(in_dir, "player_match_stats.csv"), show_col_types = FALSE,
                guess_max = 300000)
tms <- read_csv(file.path(in_dir, "team_match_stats.csv"), show_col_types = FALSE,
                guess_max = 10000)

# ---- 2. Team strength before each match -------------------------------------
team_form <- tms %>%
  arrange(team_code, kickoff_time) %>%
  group_by(team_code) %>%
  mutate(gd             = goals_for - goals_against,
         team_gd_last10 = roll_prev(gd, 10),            # goal difference per match, last 10
         rest_days      = as.numeric(difftime(kickoff_time, lag(kickoff_time), units = "days"))) %>%
  ungroup() %>%
  select(match_id, team_code, team_gd_last10, rest_days,
         any_of(c("elo_pre", "opp_elo_pre")))

opp_form <- team_form %>%
  select(match_id, opponent_code = team_code, opp_gd_last10 = team_gd_last10)

# ---- 3. Player form before each match ---------------------------------------
dc_col <- intersect(c("defensive_contribution"), names(pms))

feat <- pms %>%
  arrange(player_code, kickoff_time) %>%
  group_by(player_code) %>%
  mutate(
    played            = as.numeric(minutes > 0),
    started           = as.numeric(minutes >= 60),   # 'starts' only exists from 2022-23
    pts_last3         = roll_prev(total_points, 3),
    pts_last5         = roll_prev(total_points, 5),
    pts_last10        = roll_prev(total_points, 10),
    mins_last5        = roll_prev(minutes, 5),
    mins_prev_match   = lag(minutes),
    starts60_last5    = roll_prev_sum(started, 5),
    goals_last5       = roll_prev(goals_scored, 5),
    assists_last5     = roll_prev(assists, 5),
    bonus_last5       = roll_prev(bonus, 5),
    bps_last5         = roll_prev(bps, 5),
    ict_last5         = roll_prev(ict_index, 5),
    cs_last5          = roll_prev(clean_sheets, 5),
    xg_last5          = roll_prev(expected_goals, 5),       # FPL xG: 2022-23 onward
    xa_last5          = roll_prev(expected_assists, 5),
    defcon_last5      = if (length(dc_col)) roll_prev(.data[[dc_col]], 5) else NA_real_,
    shots_last5       = if ("core_total_shots" %in% names(pms)) roll_prev(core_total_shots, 5) else NA_real_,
    chances_last5     = if ("core_chances_created" %in% names(pms)) roll_prev(core_chances_created, 5) else NA_real_,
    matches_before    = row_number() - 1,
    played_in_last5   = roll_prev_sum(played, 5)
  ) %>%
  ungroup() %>%
  mutate(price_m = value / 10,                                # price at that gameweek, set before the deadline
         home    = as.numeric(was_home)) %>%
  left_join(team_form, by = c("match_id", "team_code")) %>%
  left_join(opp_form,  by = c("match_id", "opponent_code")) %>%
  mutate(strength_diff = team_gd_last10 - opp_gd_last10,
         elo_diff      = if ("elo_pre" %in% names(.)) elo_pre - opp_elo_pre else NA_real_,
         position      = factor(position, levels = pos_levels))

# Analysis population: players who played at least once in their previous 5 matches.
# This is decided before kickoff, so it does not leak the outcome. It drops the
# thousands of squad rows for players who never feature.
ana <- feat %>% filter(matches_before >= 5, played_in_last5 >= 1, !is.na(position))

cat(sprintf("Rows analysed: %s (from %s player-match rows)\n",
            comma(nrow(ana)), comma(nrow(feat))))

factor_labels <- c(
  pts_last3 = "Points, last 3 (mean)", pts_last5 = "Points, last 5 (mean)",
  pts_last10 = "Points, last 10 (mean)", mins_last5 = "Minutes, last 5 (mean)",
  mins_prev_match = "Minutes, previous match", starts60_last5 = "60+ min games, last 5",
  goals_last5 = "Goals, last 5 (mean)", assists_last5 = "Assists, last 5 (mean)",
  bonus_last5 = "Bonus, last 5 (mean)", bps_last5 = "BPS, last 5 (mean)",
  ict_last5 = "ICT index, last 5 (mean)", cs_last5 = "Clean sheets, last 5 (mean)",
  xg_last5 = "xG, last 5 (mean)", xa_last5 = "xA, last 5 (mean)",
  defcon_last5 = "Defensive contributions, last 5", shots_last5 = "Shots, last 5 (mean)",
  chances_last5 = "Chances created, last 5", price_m = "FPL price (GBP m)",
  home = "Playing at home", team_gd_last10 = "Own team GD/match, last 10",
  opp_gd_last10 = "Opponent GD/match, last 10", strength_diff = "Team minus opponent GD",
  elo_diff = "Elo difference (team - opp)", rest_days = "Rest days since last match"
)
factors <- intersect(names(factor_labels), names(ana))

# ---- 4. Chart 1: correlation of every factor with points, by position -------
cor_tbl <- ana %>%
  select(position, total_points, all_of(factors)) %>%
  pivot_longer(all_of(factors), names_to = "factor", values_to = "x") %>%
  filter(!is.na(x)) %>%
  group_by(position, factor) %>%
  filter(n() >= 200, sd(x) > 0) %>%
  summarise(rho = cor(x, total_points, method = "spearman"), n = n(), .groups = "drop") %>%
  mutate(label = factor_labels[factor])

write_csv(cor_tbl, file.path(out_dir, "factor_correlations.csv"))

order_by <- cor_tbl %>% group_by(label) %>% summarise(m = mean(abs(rho))) %>% arrange(m)
p1 <- cor_tbl %>%
  mutate(label = factor(label, levels = order_by$label)) %>%
  ggplot(aes(rho, label, colour = position)) +
  geom_vline(xintercept = 0, colour = "grey60") +
  geom_point(size = 2.6, alpha = .9) +
  scale_colour_manual(values = pos_cols) +
  labs(title = "Which pre-match factors move with fantasy points?",
       subtitle = "Spearman correlation with points in the match, by position. Sorted by average strength.",
       x = "Spearman correlation with points", y = NULL, colour = NULL,
       caption = "xG/xA from 2022-23; defensive contributions, shots, chances and Elo from 2025-26 only.") +
  theme_minimal(base_size = 11) + theme(legend.position = "top", panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "1_correlations_by_position.png"), p1, width = 9, height = 8, dpi = 150)

# ---- 5. Chart 2: shape of each relationship (binned means with 95% CI) -------
no_gk <- c("xg_last5", "xa_last5", "defcon_last5")   # GKs have ~0 xG/xA and no DefCon points
binned <- function(df, var, bins = 10) {
  if (var %in% no_gk) df <- filter(df, position != "GK")
  df %>% filter(!is.na(.data[[var]])) %>%
    group_by(position) %>%
    mutate(bin = ntile(.data[[var]], bins)) %>%
    group_by(position, bin) %>%
    summarise(x = mean(.data[[var]]), y = mean(total_points),
              se = sd(total_points) / sqrt(n()), n = n(), .groups = "drop") %>%
    mutate(factor = var)
}
key_numeric <- intersect(c("pts_last5", "mins_last5", "price_m", "xg_last5", "xa_last5",
                           "bonus_last5", "strength_diff", "opp_gd_last10", "defcon_last5"), factors)
bin_tbl <- bind_rows(lapply(key_numeric, \(v) binned(ana, v))) %>%
  mutate(label = factor(factor_labels[factor], levels = factor_labels[key_numeric]))

p2 <- ggplot(bin_tbl, aes(x, y, colour = position)) +
  geom_ribbon(aes(ymin = y - 1.96 * se, ymax = y + 1.96 * se, fill = position),
              alpha = .15, colour = NA) +
  geom_line(linewidth = .8) + geom_point(size = 1.4) +
  facet_wrap(~ label, scales = "free_x", ncol = 3) +
  scale_colour_manual(values = pos_cols) + scale_fill_manual(values = pos_cols) +
  labs(title = "Shape of each relationship",
       subtitle = "Factor split into 10 equal-sized groups per position; line = mean points, band = 95% CI",
       x = "Factor value (group mean)", y = "Mean fantasy points", colour = NULL, fill = NULL) +
  theme_minimal(base_size = 10) + theme(legend.position = "top", panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "2_binned_relationships.png"), p2, width = 11, height = 9, dpi = 150)

# ---- 6. Chart 3: team vs opponent strength (your core question) -------------
heat <- ana %>%
  filter(!is.na(team_gd_last10), !is.na(opp_gd_last10)) %>%
  mutate(own = cut(team_gd_last10, c(-Inf, -.5, 0, .5, 1, Inf),
                   labels = c("below -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 1", "above 1")),
         opp = cut(opp_gd_last10, c(-Inf, -.5, 0, .5, 1, Inf),
                   labels = c("below -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 1", "above 1"))) %>%
  group_by(position, own, opp) %>%
  summarise(pts = mean(total_points), n = n(), .groups = "drop") %>%
  filter(n >= 30)

p3 <- ggplot(heat, aes(opp, own, fill = pts)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.1f", pts)), size = 3) +
  facet_wrap(~ position, nrow = 1) +
  scale_fill_gradient(low = "#F1EFE8", high = "#0F6E56") +
  labs(title = "Strong team vs weak opponent: mean points per match",
       subtitle = "Strength = goal difference per match over each side's previous 10 league games (cells with at least 30 player-matches)",
       x = "Opponent strength (GD/match)", y = "Own team strength (GD/match)", fill = "Mean pts") +
  theme_minimal(base_size = 10) + theme(panel.grid = element_blank(),
                                        axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file.path(out_dir, "3_team_vs_opponent_strength.png"), p3, width = 12, height = 4.2, dpi = 150)

# ---- 7. Chart 4: playing time and home advantage ----------------------------
p4a <- ana %>%
  group_by(position, starts60_last5) %>%
  summarise(pts = mean(total_points), se = sd(total_points) / sqrt(n()), n = n(), .groups = "drop") %>%
  ggplot(aes(starts60_last5, pts, colour = position)) +
  geom_pointrange(aes(ymin = pts - 1.96 * se, ymax = pts + 1.96 * se),
                  position = position_dodge(width = .4), size = .3) +
  scale_colour_manual(values = pos_cols) +
  labs(title = "Recent regular minutes", x = "Games with 60+ minutes in last 5",
       y = "Mean fantasy points", colour = NULL) +
  theme_minimal(base_size = 10) + theme(legend.position = "top", panel.grid.minor = element_blank())

p4b <- ana %>%
  mutate(venue = if_else(home == 1, "Home", "Away")) %>%
  group_by(position, venue) %>%
  summarise(pts = mean(total_points), se = sd(total_points) / sqrt(n()), .groups = "drop") %>%
  ggplot(aes(position, pts, fill = venue)) +
  geom_col(position = position_dodge(.8), width = .7) +
  geom_errorbar(aes(ymin = pts - 1.96 * se, ymax = pts + 1.96 * se),
                position = position_dodge(.8), width = .2) +
  scale_fill_manual(values = c(Home = "#0F6E56", Away = "#B4B2A9")) +
  labs(title = "Home vs away", x = NULL, y = "Mean fantasy points", fill = NULL) +
  theme_minimal(base_size = 10) + theme(legend.position = "top", panel.grid.minor = element_blank())

p4 <- p4a + p4b + plot_layout(widths = c(1.6, 1))
ggsave(file.path(out_dir, "4_minutes_and_venue.png"), p4, width = 11, height = 4.5, dpi = 150)

# ---- 8. Chart 5: distribution of the target ---------------------------------
p5 <- ana %>%
  mutate(pts = pmin(pmax(total_points, -2), 20)) %>%
  ggplot(aes(pts, fill = position)) +
  geom_histogram(binwidth = 1, colour = "white") +
  facet_wrap(~ position, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = pos_cols, guide = "none") +
  labs(title = "The target: points per match are mostly 0 to 2 with a long right tail",
       subtitle = "Clipped to -2 to 20 for display. A model will need to handle this skew.",
       x = "Fantasy points", y = "Player-matches") +
  theme_minimal(base_size = 10) + theme(panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "5_target_distribution.png"), p5, width = 11, height = 3.6, dpi = 150)

# ---- 9. One PDF with everything ---------------------------------------------
pdf(file.path(out_dir, "factors_vs_points.pdf"), width = 11, height = 8.5)
print(p5); print(p1); print(p2); print(p3); print(p4)
invisible(dev.off())

cat("\nTop 10 factors by average |correlation| across positions:\n")
print(cor_tbl %>% group_by(label) %>% summarise(mean_rho = round(mean(rho), 3)) %>%
        arrange(desc(abs(mean_rho))) %>% head(10), n = 10)
cat("\nSaved to:", normalizePath(out_dir), "\n")