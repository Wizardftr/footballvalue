"""xG ingestion and blending.

The fixture-alignment tests matter most. Attaching one club's xG to another is a
silent corruption: nothing errors, the ratings just quietly describe the wrong
teams. The alignment is what stands between us and that, so its refusal to guess is
tested as carefully as its ability to match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fv.data.understat import align_by_fixture, parse_matches, season_to_understat
from fv.models.dixon_coles import blend_targets, fit_dixon_coles


# -- season mapping ---------------------------------------------------------

def test_season_maps_to_understat_start_year():
    assert season_to_understat("2024-25") == "2024"
    assert season_to_understat("2014-15") == "2014"


# -- payload parsing --------------------------------------------------------

def test_parse_skips_unplayed_matches():
    payload = {
        "dates": [
            {
                "id": "1", "isResult": True,
                "h": {"title": "Manchester United"}, "a": {"title": "Fulham"},
                "goals": {"h": "1", "a": "0"}, "xG": {"h": "2.04", "a": "0.41"},
                "datetime": "2024-08-16 19:00:00",
            },
            {
                "id": "2", "isResult": False,
                "h": {"title": "Arsenal"}, "a": {"title": "Chelsea"},
                "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None},
                "datetime": "2026-08-16 19:00:00",
            },
        ]
    }
    df = parse_matches(payload)
    assert len(df) == 1
    assert df.iloc[0]["home_name"] == "Manchester United"
    assert df.iloc[0]["home_xg"] == pytest.approx(2.04)


def test_parse_survives_a_malformed_record():
    payload = {"dates": [{"id": "1", "isResult": True}]}  # missing everything else
    assert parse_matches(payload).empty


# -- fixture alignment ------------------------------------------------------

def _ours(rows):
    return pd.DataFrame(
        rows, columns=["match_id", "match_date", "home_team_id", "away_team_id", "fthg", "ftag"]
    )


def _understat(rows):
    return pd.DataFrame(
        rows,
        columns=["understat_id", "kickoff", "home_name", "away_name",
                 "home_goals", "away_goals", "home_xg", "away_xg"],
    )


def test_alignment_derives_names_that_look_nothing_alike():
    """The case that motivates the whole approach.

    "Wolverhampton Wanderers" -> "Wolves" and "Nottingham Forest" -> "Nott'm Forest"
    are pairs a string matcher would plausibly get wrong. Fixture agreement gets
    them right without looking at the names at all.
    """
    ours = _ours([
        (1, "2024-08-17", 10, 20, 2, 1),
        (2, "2024-08-24", 20, 10, 0, 3),
        (3, "2024-08-31", 10, 20, 1, 1),
        (4, "2024-09-07", 20, 10, 4, 2),
    ])
    us = _understat([
        ("a", pd.Timestamp("2024-08-17 15:00"), "Wolverhampton Wanderers", "Nottingham Forest", 2, 1, 1.8, 0.9),
        ("b", pd.Timestamp("2024-08-24 15:00"), "Nottingham Forest", "Wolverhampton Wanderers", 0, 3, 0.7, 2.2),
        ("c", pd.Timestamp("2024-08-31 15:00"), "Wolverhampton Wanderers", "Nottingham Forest", 1, 1, 1.1, 1.3),
        ("d", pd.Timestamp("2024-09-07 15:00"), "Nottingham Forest", "Wolverhampton Wanderers", 4, 2, 3.1, 1.5),
    ])
    r = align_by_fixture(us, ours, min_votes=3)
    assert r.name_map["Wolverhampton Wanderers"] == 10
    assert r.name_map["Nottingham Forest"] == 20
    assert r.n_aligned == 4
    assert not r.unmatched_names


def test_alignment_tolerates_a_one_day_date_difference():
    """Understat stores local kickoff time; a late kickoff can land on the next day."""
    ours = _ours([(i, "2024-08-17", 10, 20, 2, 1) for i in range(1, 2)] +
                 [(2, "2024-08-24", 20, 10, 0, 3), (3, "2024-08-31", 10, 20, 1, 1)])
    us = _understat([
        ("a", pd.Timestamp("2024-08-18 00:30"), "Alpha", "Beta", 2, 1, 1.8, 0.9),
        ("b", pd.Timestamp("2024-08-24 15:00"), "Beta", "Alpha", 0, 3, 0.7, 2.2),
        ("c", pd.Timestamp("2024-08-31 15:00"), "Alpha", "Beta", 1, 1, 1.1, 1.3),
    ])
    r = align_by_fixture(us, ours, min_votes=3)
    assert r.name_map == {"Alpha": 10, "Beta": 20}


def test_alignment_refuses_to_guess_on_too_few_votes():
    """One agreeing fixture is a coincidence, not a mapping."""
    ours = _ours([(1, "2024-08-17", 10, 20, 2, 1)])
    us = _understat([
        ("a", pd.Timestamp("2024-08-17 15:00"), "Alpha", "Beta", 2, 1, 1.8, 0.9),
    ])
    r = align_by_fixture(us, ours, min_votes=3)
    assert r.name_map == {}
    assert r.unmatched_names == {"Alpha", "Beta"}
    assert r.n_aligned == 0


def test_alignment_rejects_an_inconsistent_mapping():
    """A name whose votes are split across teams must not be accepted."""
    ours = _ours([
        (1, "2024-08-17", 10, 20, 1, 0),
        (2, "2024-08-24", 30, 40, 1, 0),
        (3, "2024-08-31", 50, 60, 1, 0),
        (4, "2024-09-07", 70, 80, 1, 0),
    ])
    us = _understat([
        ("a", pd.Timestamp("2024-08-17 15:00"), "Ghost", "X", 1, 0, 1.0, 0.5),
        ("b", pd.Timestamp("2024-08-24 15:00"), "Ghost", "Y", 1, 0, 1.0, 0.5),
        ("c", pd.Timestamp("2024-08-31 15:00"), "Ghost", "Z", 1, 0, 1.0, 0.5),
        ("d", pd.Timestamp("2024-09-07 15:00"), "Ghost", "W", 1, 0, 1.0, 0.5),
    ])
    r = align_by_fixture(us, ours, min_votes=3, min_agreement=0.9)
    assert "Ghost" not in r.name_map
    assert "Ghost" in r.unmatched_names


def test_alignment_skips_ambiguous_scorelines():
    """Two of our matches on the same date with the same score cast no vote."""
    ours = _ours([
        (1, "2024-08-17", 10, 20, 1, 0),
        (2, "2024-08-17", 30, 40, 1, 0),  # same date, same score
    ])
    us = _understat([
        ("a", pd.Timestamp("2024-08-17 15:00"), "Alpha", "Beta", 1, 0, 1.0, 0.5),
    ])
    r = align_by_fixture(us, ours, min_votes=1)
    assert r.name_map == {}


def test_alignment_on_empty_input():
    r = align_by_fixture(pd.DataFrame(), _ours([]))
    assert r.name_map == {}
    assert r.n_aligned == 0


# -- blending ---------------------------------------------------------------

def test_blend_is_a_weighted_average():
    goals = np.array([3.0, 0.0])
    xg = np.array([1.0, 2.0])
    assert blend_targets(goals, xg, 0.5) == pytest.approx([2.0, 1.0])
    assert blend_targets(goals, xg, 0.4) == pytest.approx([2.2, 0.8])


def test_zero_weight_returns_goals_untouched():
    goals = np.array([3.0, 0.0])
    assert blend_targets(goals, np.array([1.0, 2.0]), 0.0) == pytest.approx(goals)
    assert blend_targets(goals, None, 0.5) == pytest.approx(goals)


def test_matches_without_xg_keep_their_goals():
    """Six of eleven leagues have no xG at all. Dropping those rows would shrink the
    training set and bias it toward recent, well-covered matches."""
    goals = np.array([3.0, 1.0, 2.0])
    xg = np.array([1.0, np.nan, 4.0])
    out = blend_targets(goals, xg, 0.5)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(1.0)  # untouched
    assert out[2] == pytest.approx(3.0)


def test_blending_moves_ratings_toward_the_xg_story():
    """A team that outscored its xG should rate lower once xG is blended in.

    Team A beats B 4-0 every time on 1.0 xG against 1.0 - all finishing luck. With
    goals alone A looks dominant; with xG blended in that dominance must shrink.
    """
    n = 120
    h = np.array(["A", "B"] * n)
    a = np.array(["B", "A"] * n)
    hg = np.array([4.0, 0.0] * n)
    ag = np.array([0.0, 4.0] * n)
    hxg = np.full(2 * n, 1.0)
    axg = np.full(2 * n, 1.0)
    days = np.linspace(400, 1, 2 * n)

    goals_only = fit_dixon_coles(h, a, hg, ag, days, xi=0.0)
    blended = fit_dixon_coles(h, a, hg, ag, days, xi=0.0, home_xg=hxg, away_xg=axg, xg_weight=0.5)

    a_goals = goals_only.attack[list(goals_only.teams).index("A")]
    a_blend = blended.attack[list(blended.teams).index("A")]
    assert a_blend < a_goals, "blending xG should temper a rating built on finishing luck"


def test_low_score_correction_still_uses_integer_goals():
    """The blend is continuous and would never hit 0 or 1 exactly; the tau masks must
    keep reading the real scoreline or the correction silently switches off."""
    rng = np.random.default_rng(5)
    k = 600
    teams = np.array([f"T{i}" for i in range(6)])
    h = teams[rng.integers(0, 6, k)]
    a = teams[(np.searchsorted(teams, h) + rng.integers(1, 6, k)) % 6]
    hg = rng.poisson(1.3, k).astype(float)
    ag = rng.poisson(1.1, k).astype(float)
    days = rng.uniform(1, 700, k)
    fit = fit_dixon_coles(
        h, a, hg, ag, days, xi=0.0,
        home_xg=hg + 0.3, away_xg=ag + 0.3, xg_weight=0.5,
    )
    # rho is only identifiable if the masks fired on real scorelines.
    assert fit.rho != 0.0
    assert fit.converged
