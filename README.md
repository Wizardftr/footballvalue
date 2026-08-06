# footballvalue

A personal football betting analysis tool. It models match outcomes in eleven
European leagues, compares its probabilities against bet365 prices, flags value
bets, and tracks bankroll performance.

**It never places bets.** It produces a slip you place manually. There is no
bookmaker scraping and no automated placement of any kind.

## Status: Phases 1-4 complete

| Phase | Scope | State |
|---|---|---|
| 1 | Data pipeline, SQLite schema, Dixon-Coles baseline, walk-forward backtest | **done** |
| 2 | xG ingestion, xG-blended DC, LightGBM, ensemble, market anchor | **done** |
| 3 | Streamlit dashboard, weekly slip, bet logging, auto-settlement | **done** |
| 4 | The Odds API live prices, CLV tracking, over/under 2.5 and BTTS | **done** |

## Quick start

```bash
uv sync                  # install
uv run fv init-db        # create the SQLite schema
uv run fv download       # fetch all eleven leagues, 2000-01 to date (~8 min)
uv run fv doctor         # data quality checks
uv run fv backtest       # walk-forward backtest + report in reports/

uv sync --extra model --extra xg     # Phase 2 dependencies
uv run fv download-xg                # Understat xG for the big-five divisions
uv run fv stages                     # all five model stages + comparison table

uv run fv fixtures                   # upcoming fixtures + bet365 prices
uv run fv slip --log                 # this week's slip, logged as paper bets
uv run fv settle                     # auto-settle bets whose results have arrived
uv run fv dashboard                  # Streamlit UI on localhost:8501

uv run fv markets                    # over/under 2.5 backtest + BTTS calibration
uv run fv live-odds                  # The Odds API snapshots (needs ODDS_API_KEY)
uv run fv refresh                    # the whole weekly routine in one command
```

### Weekly routine

One command does all of it:

```bash
uv run fv refresh --log-slip
```

That downloads new results and xG, promotes snapshots that are now closing prices,
backfills CLV, settles last week's bets, refreshes fixtures and live prices, and
generates the slip.

`fv backtest --help` lists the options. Useful ones:

```bash
uv run fv backtest --leagues E0,D1 --test-from 2019-08-01   # subset
uv run fv backtest --min-edge 0.06 --bankroll 2000          # override thresholds
uv run fv backtest --xi 0.002 --no-tune                     # fix the decay rate
```

## Leagues

England (Premier League, Championship), Spain (La Liga, Segunda), Italy (Serie A,
Serie B), Germany (Bundesliga, 2. Bundesliga), France (Ligue 1, Ligue 2),
Netherlands (Eredivisie). Each has an `enabled` toggle in `config.yaml`.

## Data

**Results and odds** come from [football-data.co.uk](https://www.football-data.co.uk)
CSVs. Four properties of that source shape the code, all confirmed by inspection:

- bet365 prices (`B365H/D/A`) start in **2002-03**. Earlier seasons load for their
  goals only.
- Closing prices (`B365CH/CD/CA`) start in **2019-20**. CLV is only measurable from
  then on, and the report says so rather than reporting a diluted average.
- The two price sets genuinely differ — identical on only ~14% of matches, with an
  8.3% standard deviation of movement. So the backtest bets at the pre-round price
  and measures CLV against the close. Betting *into* the closing line would
  guarantee a losing backtest and teach us nothing.
- A season that hasn't started returns an **HTML page with HTTP 200**, not a 404.
  The downloader detects this; parsing it as CSV would inject garbage.

`fv doctor` reports seasons whose match counts look wrong. Some are real gaps in the
source (E0 2003-04 and 2004-05 are missing ~45 matches each, spread through the
season); some are legitimate league-size changes (Serie B ran 22 teams until 2018).

**xG** comes from Understat, fetched directly. Two things forced that:

- The standalone `understat` package pins `pytest==7.2.0` and `pytest-cov==4.0.0` as
  *runtime* dependencies, so it cannot coexist with a test suite.
- `soccerdata`'s Understat reader depends on `tls_requests`, which downloads a native
  TLS library from GitHub releases at import time. That download is blocked here.

So `fv/data/understat.py` talks to Understat's own JSON endpoint, with local caching
and a rate limit. **FBref is unavailable** — it returns 403 to datacenter IPs
regardless of user agent — so the five non-big-five leagues plus Eredivisie run on
goals only, which is the fallback the spec allows.

Understat team names are resolved by **fixture alignment, not string similarity**.
Matches are paired on date and scoreline, and the name mapping is derived from which
pairings agree across a season. This is what correctly produced
"Wolverhampton Wanderers" → "Wolves" and "Nottingham Forest" → "Nott'm Forest",
both of which a fuzzy matcher could plausibly get wrong — and a wrong mapping
silently attaches one club's xG to another. A name that can't be established this
way is reported, not guessed. All 20,069 matches across five leagues and twelve
seasons resolved with zero unmatched names.

## Model (Phase 1)

Time-decayed Dixon-Coles bivariate Poisson on goals. Per-league attack and defence
ratings plus home advantage, with the low-score correction rho, weighted by
`exp(-xi * days)`.

- **xi is tuned** on validation seasons that end before the test window, by
  walk-forward log loss. Tuning on the test window would be leakage just as much as
  training on it.
- **Analytic gradient.** With ~90 parameters per league a numerical gradient needs
  an extra function evaluation per parameter; fits took ten seconds and still hit
  the iteration limit without converging. Differentiating by hand made them ~0.14s
  cold and ~0.02s warm, and they now converge. `tests/test_dixon_coles.py` checks
  the gradient against a numerical one.
- **Stale teams are dropped.** Under time decay a club relegated fifteen years ago
  carries a weight around 1e-7, so its ratings are unidentified and the optimiser
  never converges. Teams below a minimum effective sample are excluded, and the
  model reports that it has no opinion on them rather than inventing one.

Two things the fitting surfaced that are worth knowing: home advantage has genuinely
declined (0.27 over the full history, 0.20 on recent data), and rho has shrunk toward
zero on modern football — the 0-0 excess is still there but 1-1 now comes in *below*
independent Poisson, so the Dixon-Coles correction as specified does less than it did
in 1997.

## Model stages (Phase 2)

`fv stages` runs five stages, scores them all on the same held-out matches, and
writes a comparison. Every weight — decay rate, xG blend, ensemble pool, market
anchor — is tuned on validation seasons that **end before the test window**. Tuning
a hyperparameter on the test set is leakage just as much as training on it, and it's
the easier of the two to do by accident.

1. `dc` — time-decayed Dixon-Coles on goals (the Phase 1 baseline)
2. `dc_xg` — the same model on a goals/xG blend
3. `lgbm` — LightGBM on form, Elo, rest days and promotion flags, isotonic-calibrated
4. `ensemble` — stages 2 and 3 combined by a weighted log-opinion pool
5. `anchored` — the ensemble blended toward margin-free market probabilities

Feature building is leak-free structurally: the builder walks each league in
chronological order, writes a match's features from the state accumulated so far,
then folds that match's result into the state. There is no windowing call that could
see forward. This matters because a `groupby().rolling()` that includes the current
row leaks the result being predicted, and the model looks brilliant until it meets
real money.

LightGBM's best configuration turned out to be startlingly shallow — 4 leaves,
minimum 300 matches per leaf. With a few thousand training matches and an outcome
that is mostly noise, a 15-leaf model scored 1.015 where this one scored 0.998.

## Betting rules

Configured in `config.yaml`, all overridable per run.

- Margin stripped from bet365 prices — proportional by default, Shin's method
  available. Shin takes relatively more off longshots, which matters at the long end
  of the odds range.
- `edge = model_probability * decimal_odds - 1`
- A selection qualifies at `edge >= 0.04` with odds in `[1.50, 4.00]`.
- Stakes: 25% Kelly, capped at 2% of current bankroll, rounded **down** so rounding
  can't push a stake past its own cap.
- Singles only. Never accumulators — they multiply the bookmaker's margin.
- Max 8 bets per week, ranked by edge, across all leagues.
- Risk controls on by default: 10% weekly stop-loss, 25% drawdown pause, paper mode.

In the backtest the drawdown pause is **recorded but not enforced**, because halting
would end the evaluation at the first bad run and hide everything after it. In live
use it stops betting and requires a manual reset.

## No lookahead

Predictions for a match-week come from a model trained only on matches that kicked
off strictly before that week's Monday 00:00 UTC. The cutoff is stored on every
prediction as `trained_through`, so the property is auditable after the fact rather
than merely asserted.

`tests/test_no_lookahead.py` checks it three ways:

1. **Structurally** — every training set ends before the week it predicts.
2. **By poisoning** — rewriting every future result as 9-0 must not move any
   prediction by a single bit.
3. **By control** — poisoning the *past* must change predictions, otherwise test 2
   would only prove the code ignores its inputs.

## Honest metrics

The report leads with whether the model beats bet365's own margin-free prices on log
loss and Brier. If it doesn't, the report says so at the top.

ROI is always shown with a 95% confidence interval, because at this volume the point
estimate is misleading on its own: per-bet return standard deviation is about 1.29
units, so a 300-bet window has a standard error near 7.5 percentage points. **A
300-bet window cannot distinguish a 4% edge from zero.** That is why CLV is the
leading indicator — it resolves on every bet immediately instead of waiting for
outcomes to average out — and why monthly P&L is labelled as variance.

## Layout

```
src/fv/
  cli.py                  init-db, download, doctor, backtest
  config.py               config.yaml + .env
  db/models.py            SQLAlchemy schema
  data/football_data.py   downloader and parser
  data/teams.py           team name resolution
  data/load.py            idempotent load into SQLite
  models/dixon_coles.py   the model
  odds/                   margin, edge, kelly, settlement
  backtest/               walkforward, metrics, report
tests/
```

## The dashboard

`uv run fv dashboard`. Four pages:

- **This Week** — fixtures, model vs bet365, edge per selection, the recommended slip
  with stakes, and text/CSV export for placing by hand. It also lists every selection
  that *didn't* qualify with the reason, so "why is this not on the slip?" is
  answerable.
- **Backtest** — stage comparison, cumulative P&L, drawdown, ROI by league and season,
  CLV distribution with its confidence interval.
- **Bankroll** — balance, full bet log, paper vs real split, rolling 300-bet ROI *with
  its confidence band*, monthly P&L labelled as variance, longest losing streak.
- **Settings** — bankroll, Kelly fraction, edge threshold, odds range, league toggles,
  stop-loss levels, and the paper/real toggle.

### The real-money gate

Paper mode is on by default and real money is *earned*, not assumed. The Settings
page computes the verdict from stored results rather than from anyone's
recollection, and shows it beside the toggle:

1. the model must beat bet365's closing prices out-of-sample in `fv stages`, and
2. paper trading must have run at least 4 weeks with CLV significantly above zero.

An absent backtest counts as a failure, not an unknown — a gate you can pass by not
running the test is not a gate. **As of the Phase 2 results the gate correctly
refuses**: the model is 0.34 millinats behind bet365, and no paper trading has run.

Both `fv slip --real` and the dashboard toggle still let you override it. They just
make you look at the evidence first.

## Markets

**1X2** is the v1 market and the one with the most evidence behind it.

**Over/under 2.5** is fully backtested. bet365 published these prices in
football-data from 2019-20, pre and closing, so it gets the same treatment as 1X2 —
walk-forward predictions, real prices, real CLV. It was worth testing on the theory
that it might be softer than 1X2. **It isn't** — see the results section.

**BTTS is not validated.** There are no historical BTTS prices in football-data or
any other free source this project can reach, so there is no edge, ROI or CLV to
compute and no backtest is possible. The model produces a BTTS probability and its
calibration is checked, but under the project's own rule — an upgrade ships only if
it beats the previous stage out-of-sample — BTTS must not be bet on. Prices only
begin accumulating once The Odds API snapshots start. The calibration check also
found a real problem: the model **under-predicts BTTS by about 1.9 percentage
points**, which would bias it toward backing "No".

All three come from the same fitted Dixon-Coles score matrix rather than separate
models — they are different aggregations of one joint distribution over scorelines,
so they can never contradict each other.

## The Odds API (Phase 4)

Used for what football-data cannot provide: BTTS prices, and repeated snapshots near
kickoff so CLV can be measured on bets actually placed.

Put a key in `.env` (never committed). Without one, `fv refresh` skips the live-price
step and carries on — football-data still covers 1X2 and over/under 2.5.

Quota is the binding constraint: the free tier allows 500 requests a month and one
request costs `regions x markets` credits, not one. So markets are requested together
in a single call per league, responses are cached, and the remaining quota is read
back from the response headers and reported.

Closing prices are **derived** from snapshots rather than trusted from any single
fetch. Only a snapshot taken within 12 hours of kickoff is promoted to a closing
price. A price captured four days out is not the close, and treating it as one would
inflate CLV — the metric the whole project rests on.

## Results

Every stage and market, measured against bet365's own margin-free prices on held-out
matches:

| market / stage | model log loss | bet365 | verdict |
|---|---:|---:|---|
| 1X2 — Dixon-Coles (goals) | 1.02326 | 1.00255 | behind |
| 1X2 — + xG blend | 1.02149 | 1.00255 | behind |
| 1X2 — + LightGBM ensemble | 1.02091 | 1.00255 | behind |
| 1X2 — market-anchored | 1.00289 | 1.00255 | **behind by 0.34 millinats** |
| Over/under 2.5 | 0.68482 | 0.67418 | behind by 10.6 millinats |
| BTTS | 0.69061 | — | **no prices exist; unvalidated** |

**Nothing beats the market.** ROI is negative on both tradeable markets (1X2 -8.1%,
over/under -6.8%) and CLV is statistically indistinguishable from zero on both. The
dashboard says so, and the real-money gate refuses on that basis.

The most telling number is in the tuned weights: **eight of eleven leagues tuned the
market anchor to 1.0**, meaning validation's own verdict was to ignore the model
entirely and use the price. That is what a genuine absence of edge looks like when
you measure it honestly instead of hoping.

## Ground rules

- No scraping of bet365 or any bookmaker site. No automated bet placement.
- Tests are required for all odds maths: margin removal, edge, Kelly, settlement.
- `.env` holds the Odds API key and is never committed.
