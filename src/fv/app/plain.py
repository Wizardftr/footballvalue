"""Plain English.

Everything the user reads goes through here. The maths is unchanged — this module
only decides what the numbers are *called*, and it is a separate file so that
renaming something in the interface can never accidentally rename something in the
model.

The rule: a word earns its place only if somebody who has never read a betting
forum would understand it. "Edge", "Kelly fraction", "closing line value", "log
loss" and "walk-forward" all fail that test. Where the honest version of a
sentence needs a technical idea, the idea is explained rather than named.
"""

from __future__ import annotations

PICK_WORDS = {
    "H": "Home win", "D": "Draw", "A": "Away win",
    "O": "Over 2.5 goals", "U": "Under 2.5 goals",
}

MARKET_WORDS = {"1X2": "Who wins", "OU25": "Goals"}

# Model stages, as the Track record page names them.
STAGE_NAMES = {
    "dc": "Goals only",
    "dc_xg": "Goals + chances created",
    "lgbm": "Recent form",
    "ensemble": "Everything combined",
    "anchored": "Combined, checked against the bookmaker",
}

# Staking styles. Each is a complete set of the numbers the old Settings page made
# people choose one by one. The values are unchanged — only the way they are picked.
STAKING_STYLES: dict[str, dict] = {
    "Careful": {
        "kelly_fraction": 0.10,
        "max_stake_pct": 0.01,
        "min_edge": 0.06,
        "blurb": "Small stakes, and only the picks the model likes most. "
                 "Fewer picks, slower swings.",
    },
    "Balanced": {
        "kelly_fraction": 0.25,
        "max_stake_pct": 0.02,
        "min_edge": 0.04,
        "blurb": "The default. Never more than 2% of your balance on one match.",
    },
    "Bold": {
        "kelly_fraction": 0.50,
        "max_stake_pct": 0.04,
        "min_edge": 0.02,
        "blurb": "Bigger stakes and a looser filter. Expect much larger ups and "
                 "downs, including long losing runs.",
    },
}


def style_of(settings: dict) -> str:
    """Which style the current numbers correspond to, or 'Custom'."""
    for name, preset in STAKING_STYLES.items():
        if all(
            abs(float(settings.get(key, -1)) - value) < 1e-9
            for key, value in preset.items()
            if key != "blurb"
        ):
            return name
    return "Custom"


GLOSSARY = [
    ("Stake", "How much money to put on one match."),
    ("Odds", "What the bookmaker pays. Odds of 2.50 turn a €10 stake into €25 if it "
             "wins — your €10 back plus €15 profit."),
    ("Our chance / Their chance",
     "How likely the model thinks a result is, next to how likely the bookmaker's "
     "price says it is. A pick is suggested when the model thinks something is more "
     "likely than the price implies."),
    ("Value", "The gap between those two numbers. It is the only reason a match ends "
              "up on your list — not because a team looks good, but because the price "
              "looks wrong."),
    ("Practice mode", "The app records your picks and results without any real money "
                      "being involved, so you can see how it would have gone."),
    ("Balance", "Your running total in the app. It starts at whatever you set and "
                "moves as results come in."),
    ("Return", "Profit divided by everything you staked. +5% means you got back €1.05 "
               "for every €1 risked."),
    ("Closing price", "The bookmaker's final price just before kick-off. If you "
                      "regularly take a better price than that, it is a sign you are "
                      "spotting something early. It is the best early warning there "
                      "is, long before profit shows up."),
    ("Over / Under 2.5 goals",
     "A bet on the total scored by both teams. Over 2.5 wins if there are three or "
     "more goals; under 2.5 wins if there are two or fewer. It has nothing to do "
     "with who wins, which is why it is often the more predictable question."),
    ("Singles only", "One bet per match, never combined. Combining several picks into "
                     "one bet multiplies the bookmaker's cut and is the fastest way to "
                     "lose money."),
]


def readiness_lines(r) -> list[str]:
    """The real-money verdict, in plain sentences rather than model diagnostics."""
    lines = []
    if r.backtest_beats_closing:
        lines.append("✅ In testing on past seasons, the model priced matches better "
                     "than the bookmaker did.")
    else:
        lines.append("❌ In testing on past seasons, the model did **not** price "
                     "matches better than the bookmaker.")

    weeks = r.paper_weeks
    if weeks >= 4:
        lines.append(f"✅ You have {weeks:.0f} weeks of practice results ({r.paper_bets} picks).")
    else:
        lines.append(f"❌ Only {weeks:.1f} of the 4 weeks of practice are done "
                     f"({r.paper_bets} picks so far).")

    if r.paper_clv_significant:
        lines.append("✅ Your practice picks consistently beat the bookmaker's final price.")
    elif r.paper_bets:
        lines.append("❌ Your practice picks have not beaten the bookmaker's final "
                     "price often enough to mean anything yet.")
    else:
        lines.append("❌ No practice results yet to compare against the bookmaker's "
                     "final price.")
    return lines
