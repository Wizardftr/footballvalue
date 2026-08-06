"""Command line interface.

    uv run fv init-db                 create the SQLite schema
    uv run fv download                fetch every enabled league-season
    uv run fv doctor                  data quality checks
    uv run fv backtest                walk-forward backtest + report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from fv.config import PROJECT_ROOT, load_config
from fv.db.session import init_db as _init_db

app = typer.Typer(add_completion=False, help="Football betting value analysis.")
console = Console()


@app.command("init-db")
def init_db_cmd():
    """Create the database schema and sync league reference rows."""
    path = _init_db()
    console.print(f"[green]database ready[/green] {path}")


@app.command()
def download(
    leagues: str = typer.Option(None, help="Comma-separated league codes (default: all enabled)."),
    first_season: str = typer.Option(None, help='Earliest season, e.g. "2000-01".'),
    force: bool = typer.Option(False, help="Re-parse files even when unchanged."),
):
    """Download football-data.co.uk results and bet365 odds, then load into SQLite."""
    from fv.data.load import download_and_load

    cfg = load_config()
    _init_db(cfg)
    codes = [c.strip() for c in leagues.split(",")] if leagues else None
    stats = download_and_load(cfg, leagues=codes, first_season=first_season, force=force)
    console.print(f"[green]{stats.summary()}[/green]")
    for err in stats.errors[:20]:
        console.print(f"[red]  {err}[/red]")


@app.command("download-xg")
def download_xg(
    leagues: str = typer.Option(None, help="Comma-separated league codes."),
    first_season: str = typer.Option("2014-15", help="Earliest season (Understat starts 2014-15)."),
    force: bool = typer.Option(False, help="Re-fetch even when cached."),
):
    """Download Understat xG for the big-five top divisions.

    The other six leagues have no reliable free xG source and run on goals only.
    """
    from fv.data.understat import LEAGUE_SLUGS, download_and_load_xg

    cfg = load_config()
    _init_db(cfg)
    codes = [c.strip() for c in leagues.split(",")] if leagues else None
    console.print(f"[cyan]Understat covers: {', '.join(LEAGUE_SLUGS)}[/cyan]")

    with console.status("fetching") as status:
        stats = download_and_load_xg(
            cfg,
            leagues=codes,
            first_season=first_season,
            force=force,
            progress=lambda c, s: status.update(f"{c} {s}"),
        )

    console.print(f"[green]{stats.summary()}[/green]")
    for code, names in stats.unmatched_names.items():
        console.print(f"[yellow]{code}: {len(names)} team name(s) not resolved by fixture "
                      f"alignment: {sorted(names)[:8]}[/yellow]")


@app.command()
def doctor():
    """Data quality checks. Surfaces gaps rather than letting them pass silently."""
    from sqlalchemy import text

    from fv.db.session import get_engine

    cfg = load_config()
    eng = get_engine(cfg)

    with eng.connect() as conn:
        matches = pd.read_sql(
            text("SELECT league_code, season, COUNT(*) n FROM matches GROUP BY 1,2"), conn
        )
        odds = pd.read_sql(
            text("""SELECT m.league_code, m.season,
                       SUM(CASE WHEN o.odds_type='pre' THEN 1 ELSE 0 END) pre,
                       SUM(CASE WHEN o.odds_type='closing' THEN 1 ELSE 0 END) closing
                    FROM matches m LEFT JOIN odds o ON o.match_id=m.id
                    GROUP BY 1,2"""), conn
        )
        teams = pd.read_sql(text("SELECT country, COUNT(*) n FROM teams GROUP BY 1"), conn)

    if matches.empty:
        console.print("[red]no matches loaded - run `fv download` first[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{matches.n.sum():,} matches across "
                  f"{matches.league_code.nunique()} leagues[/bold]")

    t = Table(title="teams by country")
    t.add_column("country"); t.add_column("teams", justify="right")
    for r in teams.itertuples(index=False):
        t.add_row(r.country, f"{r.n:,}")
    console.print(t)

    # A season whose match count is far from its league's usual count is a data gap.
    console.print("\n[bold]seasons with anomalous match counts[/bold]")
    anomalies = []
    for code, g in matches.groupby("league_code"):
        typical = g.n.median()
        for r in g.itertuples(index=False):
            if abs(r.n - typical) > 0.05 * typical:
                anomalies.append((code, r.season, r.n, typical))
    if anomalies:
        t = Table()
        for c in ("league", "season", "matches", "typical"):
            t.add_column(c, justify="right" if c != "league" else "left")
        for code, season, n, typ in sorted(anomalies):
            t.add_row(code, season, str(n), f"{typ:.0f}")
        console.print(t)
        console.print("[yellow]These are gaps in the football-data source files, not "
                      "parse errors. Verified for E0 2003-04 and 2004-05.[/yellow]")
    else:
        console.print("  none")

    merged = matches.merge(odds, on=["league_code", "season"], how="left")
    no_odds = merged[merged.pre.fillna(0) == 0]
    console.print(f"\n[bold]seasons with no bet365 prices:[/bold] {len(no_odds)} "
                  "(expected: everything before 2002-03)")
    no_close = merged[(merged.pre.fillna(0) > 0) & (merged.closing.fillna(0) == 0)]
    console.print(f"[bold]seasons with prices but no closing prices:[/bold] {len(no_close)} "
                  "(expected: everything before 2019-20 - CLV is unmeasurable there)")


@app.command()
def backtest(
    leagues: str = typer.Option(None, help="Comma-separated league codes."),
    test_from: str = typer.Option(None, help='Start of the test window, e.g. "2015-08-01".'),
    test_to: str = typer.Option(None, help="End of the test window."),
    xi: float = typer.Option(None, help="Time-decay rate. Omit to tune on validation seasons."),
    tune: bool = typer.Option(True, help="Tune xi on validation seasons before the test window."),
    out: Path = typer.Option(None, help="Report directory (default: reports/)."),
    bankroll: float = typer.Option(None, help="Starting bankroll."),
    min_edge: float = typer.Option(None, help="Minimum edge to bet."),
    flat_stake: float = typer.Option(
        None,
        help="Bet a fixed amount instead of Kelly. Measures the edge itself without "
        "the compounding path, and keeps late seasons in the sample.",
    ),
):
    """Run the walk-forward backtest and write a report."""
    from fv.backtest.report import build_report, write_report
    from fv.backtest.walkforward import (
        BettingParams,
        generate_predictions,
        load_matches,
        simulate_betting,
        tune_xi,
    )
    from fv.odds.edge import Thresholds
    from fv.odds.kelly import StakeRules

    cfg = load_config()
    codes = (
        [c.strip() for c in leagues.split(",")]
        if leagues
        else [lg.code for lg in cfg.enabled_leagues]
    )
    bt = cfg.backtest
    start = test_from or bt.get("test_from", "2010-11")
    if len(start) == 7 and "-" in start:  # a season label like "2015-16"
        start = f"{start.split('-')[0]}-08-01"
    end = test_to or bt.get("test_to")

    dc = cfg.dixon_coles
    grid = dc.get("xi_grid", [0.0005, 0.001, 0.002, 0.003])

    all_preds, all_diags = [], []
    xi_tables = []

    for code in codes:
        matches = load_matches(code, cfg)
        if matches.empty:
            console.print(f"[yellow]{code}: no data, skipping[/yellow]")
            continue

        chosen_xi = xi if xi is not None else dc.get("xi")
        table = None
        if chosen_xi is None and tune:
            # Validation window: the three seasons immediately before the test start.
            v_end = pd.Timestamp(start) - pd.Timedelta(days=1)
            v_start = v_end - pd.DateOffset(years=3)
            console.print(f"[cyan]{code}: tuning xi on {v_start.date()} to {v_end.date()}[/cyan]")
            chosen_xi, table = tune_xi(code, v_start, v_end, grid, cfg=cfg, matches=matches)
            if table is not None and not table.empty:
                table = table.assign(league_code=code)
                xi_tables.append(table)
        chosen_xi = chosen_xi if chosen_xi is not None else 0.0018

        console.print(f"[cyan]{code}: walk-forward from {start} (xi={chosen_xi:g})[/cyan]")
        result = generate_predictions(code, start, end, xi=chosen_xi, cfg=cfg, matches=matches)
        if not result.predictions.empty:
            all_preds.append(result.predictions)
        if not result.diagnostics.empty:
            all_diags.append(result.diagnostics)

    if not all_preds:
        console.print("[red]no predictions produced[/red]")
        raise typer.Exit(1)

    predictions = pd.concat(all_preds, ignore_index=True).sort_values("kickoff_utc")
    diagnostics = pd.concat(all_diags, ignore_index=True) if all_diags else pd.DataFrame()

    betting = cfg.betting
    risk = cfg.risk
    params = BettingParams(
        starting_bankroll=bankroll or betting.get("starting_bankroll", 1000.0),
        thresholds=Thresholds(
            min_edge=min_edge if min_edge is not None else betting.get("min_edge", 0.04),
            min_odds=betting.get("min_odds", 1.50),
            max_odds=betting.get("max_odds", 4.00),
        ),
        stake_rules=StakeRules(
            kelly_fraction=betting.get("kelly_fraction", 0.25),
            max_stake_pct=betting.get("max_stake_pct", 0.02),
            rounding=betting.get("stake_rounding", 0.50),
            min_stake=betting.get("min_stake", 1.00),
        ),
        max_bets_per_week=betting.get("max_bets_per_week", 8),
        min_team_matches=cfg.dixon_coles.get("min_team_matches", 6),
        margin_method=cfg.odds.get("margin_method", "proportional"),
        # Flat staking is an evaluation mode: risk controls keyed to a shrinking
        # bankroll would truncate the sample, which is exactly what it exists to avoid.
        weekly_stop_loss_pct=None if flat_stake else risk.get("weekly_stop_loss_pct", 0.10),
        max_drawdown_pct=risk.get("max_drawdown_pct", 0.25),
        flat_stake=flat_stake,
    )

    console.print(f"[cyan]simulating {len(predictions):,} predictions[/cyan]")
    sim = simulate_betting(predictions, params)

    xi_table = pd.concat(xi_tables, ignore_index=True) if xi_tables else None
    report = build_report(
        predictions=predictions,
        bets=sim.bets,
        equity=sim.equity,
        params={
            "starting_bankroll": params.starting_bankroll,
            "flat_stake": params.flat_stake,
        },
        diagnostics=diagnostics,
        halted_at=sim.halted_at,
        xi_table=xi_table,
        drawdown_breaches=sim.drawdown_breaches,
    )

    out_dir = out or (PROJECT_ROOT / "reports")
    path = write_report(out_dir, report, predictions, sim.bets, sim.equity)
    console.print(f"\n[green]report written to {path}[/green]\n")
    console.print(report)


@app.command()
def stages(
    leagues: str = typer.Option(None, help="Comma-separated league codes."),
    valid_from: str = typer.Option("2019-08-01", help="Start of the validation window."),
    test_from: str = typer.Option("2022-08-01", help="Start of the test window."),
    test_to: str = typer.Option(None, help="End of the test window."),
    flat_stake: float = typer.Option(10.0, help="Flat stake for the betting comparison."),
    out: Path = typer.Option(None, help="Report directory (default: reports/stages)."),
):
    """Run all five model stages and produce the stage-by-stage comparison.

    Every weight is tuned on the validation window, which ends before the test
    window begins, so no stage sees test data while being configured.
    """
    from fv.backtest.stage_runner import run_stages

    cfg = load_config()
    codes = (
        [c.strip() for c in leagues.split(",")]
        if leagues
        else [lg.code for lg in cfg.enabled_leagues]
    )
    out_dir = out or (PROJECT_ROOT / "reports" / "stages")
    report = run_stages(
        cfg,
        codes,
        valid_from=valid_from,
        test_from=test_from,
        test_to=test_to,
        flat_stake=flat_stake,
        out_dir=out_dir,
        console=console,
    )
    console.print(report)


@app.command()
def fixtures():
    """Download upcoming fixtures and their bet365 prices."""
    from fv.data.fixtures import download_fixtures

    cfg = load_config()
    _init_db(cfg)
    stats = download_fixtures(cfg)
    console.print(f"[green]{stats.summary()}[/green]")
    if stats.in_scope == 0:
        console.print("[yellow]No fixtures for enabled leagues. football-data publishes "
                      "about a week ahead and only once a season is under way.[/yellow]")
    for name in stats.unresolved[:10]:
        console.print(f"[red]unresolved team name: {name}[/red]")


@app.command()
def slip(
    bankroll: float = typer.Option(None, help="Override the ledger bankroll."),
    log: bool = typer.Option(False, "--log", help="Record the slip as pending bets."),
    real: bool = typer.Option(False, "--real", help="Log as real money instead of paper."),
    out: Path = typer.Option(None, help="Write the slip to a file."),
):
    """Generate this week's recommended slip."""
    from fv.bets import current_bankroll, log_slip
    from fv.slip import generate_slip, slip_to_csv, slip_to_text

    cfg = load_config()
    br = bankroll if bankroll is not None else current_bankroll(cfg)
    result = generate_slip(cfg, bankroll=br)
    text_slip = slip_to_text(result)
    console.print(text_slip)

    if result.untuned_leagues:
        console.print(f"[yellow]No tuned weights for {', '.join(result.untuned_leagues)}; "
                      "using config defaults. Run `fv stages` first.[/yellow]")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text_slip)
        out.with_suffix(".csv").write_text(slip_to_csv(result))
        console.print(f"[green]written to {out} and {out.with_suffix('.csv')}[/green]")

    if log and not result.selections.empty:
        if real:
            from fv.settings_store import real_money_readiness

            r = real_money_readiness(cfg)
            if not r.ready:
                console.print(f"[red]{r.headline}[/red]")
                for reason in r.reasons:
                    console.print(f"[red]  - {reason}[/red]")
                if not typer.confirm("Log these as REAL MONEY anyway?"):
                    raise typer.Abort()
        slip_id, n = log_slip(result.selections, mode="real" if real else "paper", cfg=cfg)
        console.print(f"[green]logged {n} bets as {slip_id}[/green]")


@app.command()
def settle():
    """Auto-settle pending bets whose results have arrived."""
    from fv.bets import settle_pending

    stats = settle_pending(load_config())
    console.print(f"[green]{stats.summary()}[/green]")


@app.command()
def dashboard(port: int = typer.Option(8501, help="Port to serve on.")):
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "app" / "dashboard.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.port", str(port), "--server.headless", "true"],
        check=False,
    )


@app.command("live-odds")
def live_odds(
    leagues: str = typer.Option(None, help="Comma-separated league codes."),
    force: bool = typer.Option(False, help="Ignore the cache and re-fetch."),
):
    """Fetch current prices from The Odds API and store them as snapshots."""
    from fv.data.live_odds import store_snapshots
    from fv.data.odds_api import MissingApiKey

    cfg = load_config()
    if not cfg.odds_api_key:
        console.print("[yellow]ODDS_API_KEY is not set.[/yellow] Copy .env.example to .env "
                      "and add a key from https://the-odds-api.com (the free tier is enough "
                      "for weekly use). Until then football-data prices are used, which cover "
                      "1X2 and over/under 2.5 but not BTTS.")
        raise typer.Exit(1)

    codes = [c.strip() for c in leagues.split(",")] if leagues else None
    try:
        stats = store_snapshots(cfg, leagues=codes, force=force)
    except MissingApiKey as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[green]{stats.summary()}[/green]")
    if stats.unresolved_names:
        console.print(f"[yellow]{len(stats.unresolved_names)} team name(s) unresolved: "
                      f"{sorted(stats.unresolved_names)[:8]}[/yellow]")
    for e in stats.unmatched_events[:5]:
        console.print(f"[yellow]  no fixture matched: {e}[/yellow]")


@app.command()
def refresh(
    skip_download: bool = typer.Option(False, help="Skip the results download."),
    skip_xg: bool = typer.Option(False, help="Skip the xG download."),
    log_slip: bool = typer.Option(False, "--log-slip", help="Log the generated slip."),
):
    """The weekly routine, in one command.

    Downloads new results, settles last week's bets, refreshes fixtures and prices,
    promotes closing odds from snapshots, backfills CLV, and generates the slip.
    """
    from fv.bets import current_bankroll, log_slip as do_log_slip, settle_pending
    from fv.data.fixtures import download_fixtures
    from fv.data.live_odds import backfill_clv, promote_closing_odds, store_snapshots
    from fv.slip import generate_slip, slip_to_text

    cfg = load_config()
    _init_db(cfg)
    step = 0

    def heading(text: str):
        nonlocal step
        step += 1
        console.rule(f"[bold cyan]{step}. {text}")

    if not skip_download:
        heading("New results")
        from fv.data.load import download_and_load
        console.print(download_and_load(cfg, first_season="2024-25").summary())

    if not skip_xg:
        heading("xG")
        from fv.data.understat import download_and_load_xg
        console.print(download_and_load_xg(cfg, first_season="2024-25").summary())

    heading("Closing prices and CLV")
    # Snapshots become closing prices once their match has kicked off.
    console.print(promote_closing_odds(cfg).summary())
    filled = backfill_clv(cfg)
    console.print(f"CLV backfilled on {filled} settled bets")

    heading("Settle last week")
    console.print(settle_pending(cfg).summary())

    heading("Upcoming fixtures")
    console.print(download_fixtures(cfg).summary())

    if cfg.odds_api_key:
        heading("Live prices")
        try:
            console.print(store_snapshots(cfg).summary())
        except Exception as exc:
            console.print(f"[yellow]live odds unavailable: {exc}[/yellow]")
    else:
        console.print("[dim]ODDS_API_KEY not set - skipping live prices "
                      "(football-data prices still cover 1X2 and over/under 2.5)[/dim]")

    heading("This week's slip")
    result = generate_slip(cfg, bankroll=current_bankroll(cfg))
    console.print(slip_to_text(result))
    if log_slip and not result.selections.empty:
        slip_id, n = do_log_slip(result.selections, mode="paper", cfg=cfg)
        console.print(f"[green]logged {n} bets as {slip_id}[/green]")


@app.command()
def markets(
    leagues: str = typer.Option(None, help="Comma-separated league codes."),
    test_from: str = typer.Option("2022-08-01", help="Start of the test window."),
    out: Path = typer.Option(None, help="Report directory (default: reports/markets)."),
):
    """Backtest over/under 2.5 and check BTTS calibration."""
    from fv.backtest.market_report import run_market_backtest

    cfg = load_config()
    codes = ([c.strip() for c in leagues.split(",")] if leagues
             else [lg.code for lg in cfg.enabled_leagues])
    report = run_market_backtest(cfg, codes, test_from=test_from,
                                 out_dir=out or (PROJECT_ROOT / "reports" / "markets"),
                                 console=console)
    console.print(report)


if __name__ == "__main__":
    app()
