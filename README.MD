# Fantasy Points Prediction Model
## Plan
We will be using 2 models for comparison
### 1st Model (baseline)
The average fantasy points from last matches
### 2nd Model

Inputs:
| Factor | Example input |
|---|---|
| Team strength | Team Elo rating, league position, goal difference |
| Opponent strength | Opponent Elo, goals conceded, expected goals against |
| Recent form | Player's last 5-match and last 10-match statistics |
| Attacking ability | xG, xA, shots, chances created per 90 |
| Defensive ability | Qualifying defensive contributions per 90 |
| Playing time | Probability of starting, expected minutes |
| Tactical matchup | Team possession, opponent defensive style |
| Set pieces | Penalty taker, corners, free kicks |
| Match conditions | Home/away, rest days, fixture congestion |
| Player role | Position, starting role, recent tactical changes |

### Comparison Metrics
**MAE (mean absolute error)**: how many fantasy points your predictions miss by on average.
**RMSE**: penalises large prediction errors more heavily.
**Calibration**: when the model assigns probabilities to outcomes, check whether those probabilities match what happens over many matches.

**Baseline improvement**: compare against recent average points and other simple predictions.

## Execution
### Stage 1 — R: Data science and analysis
Clean and explore historical football data.
Visualise relationships between team strength, opponent strength and fantasy points.
Analyse player form and home/away effects.
Test hypotheses with regression and statistical models.
Identify useful features for the predictive model.
Packages: ***tidyverse, ggplot2, tidymodels, lme4***

What to do
**Analysis A**: Team strength vs fantasy points
Does playing for a stronger team actually lead to higher fantasy points? Compare team ratings with average player points, separating goalkeepers, defenders, midfielders and attackers.

**Analysis B**: Recent form vs future performance
Compare the last 3, 5 and 10 matches of player statistics against their next-match fantasy points. This helps identify useful form windows.

**Analysis C**: Matchup effects
Measure how player points change against stronger or weaker opponents. Separate attacking and defensive roles instead of assuming every player benefits equally from an easy fixture.

**Analysis D**: Minutes and point distribution
Determine how much expected playing time explains fantasy scores. Investigate whether players with similar per-90-minute statistics have different actual points because of rotation or substitution risk.

### Stage 2 — Python: Data pipeline and modelling
- Load and preprocess your training data.
- Create features for each upcoming fixture.
- Train and evaluate a baseline model.
- Build and train a TensorFlow neural network.
- Generate predictions for upcoming matches.
Libraries: ***pandas, scikit-learn, tensorflow, matplotlib***

### Shared Data Layer
- Store your cleaned player-match data in CSV or Parquet files so both R and Python can use the same dataset.
- Keep a consistent player ID, match ID, date and scoring system.



