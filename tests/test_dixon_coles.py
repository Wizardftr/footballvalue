"""Dixon-Coles fitting.

The headline test simulates matches from known ratings and checks the fitter
recovers them. Without that, a subtly wrong likelihood or gradient would still
produce plausible-looking numbers and nothing would catch it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import check_grad

from fv.models.dixon_coles import (
    _nll_and_gradient,
    _trim_stale_teams,
    fit_dixon_coles,
)


def _simulate(n_teams=14, n_rounds=60, home_adv=0.28, seed=11):
    """Generate matches from known attack/defence ratings."""
    rng = np.random.default_rng(seed)
    attack = rng.normal(0, 0.30, n_teams)
    attack -= attack.mean()  # matches the identifiability constraint
    defence = rng.normal(0, 0.25, n_teams)

    h, a, hg, ag, days = [], [], [], [], []
    for r in range(n_rounds):
        order = rng.permutation(n_teams)
        for i in range(0, n_teams - 1, 2):
            ti, tj = order[i], order[i + 1]
            lam = np.exp(attack[ti] + defence[tj] + home_adv)
            mu = np.exp(attack[tj] + defence[ti])
            h.append(ti)
            a.append(tj)
            hg.append(rng.poisson(lam))
            ag.append(rng.poisson(mu))
            days.append((n_rounds - r) * 7)
    names = np.array([f"T{i:02d}" for i in range(n_teams)])
    return (
        names[np.array(h)], names[np.array(a)],
        np.array(hg, float), np.array(ag, float), np.array(days, float),
        attack, defence, names,
    )


def test_analytic_gradient_matches_numerical():
    """A wrong gradient produces a wrong fit that still looks reasonable."""
    rng = np.random.default_rng(3)
    n, k = 8, 500
    hi = rng.integers(0, n, k)
    ai = (hi + rng.integers(1, n, k)) % n
    hg = rng.poisson(1.5, k).astype(float)
    ag = rng.poisson(1.1, k).astype(float)
    w = np.exp(-0.002 * rng.uniform(0, 900, k))
    x0 = rng.normal(0, 0.3, 2 * n + 1)
    x0[2 * n] = -0.08

    err = check_grad(
        lambda p: _nll_and_gradient(p, n, hi, ai, hg, ag, w)[0],
        lambda p: _nll_and_gradient(p, n, hi, ai, hg, ag, w)[1],
        x0,
    )
    assert err < 1e-4, f"analytic gradient disagrees with numerical (err={err})"


def test_recovers_known_ratings_from_simulated_data():
    h, a, hg, ag, days, attack, defence, names = _simulate(n_rounds=200)
    # No time decay: every simulated match came from the same true ratings.
    fit = fit_dixon_coles(h, a, hg, ag, days, xi=0.0)

    order = [list(fit.teams).index(n_) for n_ in names]
    est_attack = fit.attack[order]
    est_defence = fit.defence[order]

    assert np.corrcoef(est_attack, attack)[0, 1] > 0.95
    assert np.corrcoef(est_defence, defence)[0, 1] > 0.95
    assert fit.home_adv == pytest.approx(0.28, abs=0.06)
    assert fit.converged


def test_attack_ratings_sum_to_zero():
    """The identifiability constraint must hold by construction."""
    h, a, hg, ag, days, *_ = _simulate()
    fit = fit_dixon_coles(h, a, hg, ag, days, xi=0.0)
    assert fit.attack.sum() == pytest.approx(0.0, abs=1e-9)


def test_predictions_are_a_probability_distribution():
    h, a, hg, ag, days, *_ = _simulate()
    fit = fit_dixon_coles(h, a, hg, ag, days, xi=0.001)
    for home in fit.teams[:4]:
        for away in fit.teams[:4]:
            if home == away:
                continue
            p = fit.predict(home, away)
            assert sum(p) == pytest.approx(1.0, abs=1e-9)
            assert all(0.0 <= x <= 1.0 for x in p)


def test_home_advantage_favours_the_home_side():
    """The same pairing must be more likely to be won at home than away."""
    h, a, hg, ag, days, *_ = _simulate()
    fit = fit_dixon_coles(h, a, hg, ag, days, xi=0.0)
    x, y = fit.teams[0], fit.teams[1]
    p_home_when_home = fit.predict(x, y)[0]
    p_away_when_away = fit.predict(y, x)[2]
    assert p_home_when_home > p_away_when_away


def test_time_decay_weights_recent_matches_more():
    """A team that was bad and became good should be rated better under decay.

    Both blocks are played home and away so the change in A's rating can't be
    confounded with home advantage.
    """
    n = 200
    h, a, hg, ag, days = [], [], [], [], []

    def add(home, away, home_goals, away_goals, day):
        h.append(home); a.append(away); hg.append(home_goals); ag.append(away_goals)
        days.append(day)

    # Old block: A is thrashed 0-4 both home and away.
    for d in np.linspace(1500, 800, n):
        add("A", "B", 0.0, 4.0, d)
        add("B", "A", 4.0, 0.0, d)
    # Recent block: A thrashes B 4-0 both home and away.
    for d in np.linspace(700, 1, n):
        add("A", "B", 4.0, 0.0, d)
        add("B", "A", 0.0, 4.0, d)

    h, a = np.array(h), np.array(a)
    hg, ag, days = np.array(hg), np.array(ag), np.array(days)

    flat = fit_dixon_coles(h, a, hg, ag, days, xi=0.0)
    decayed = fit_dixon_coles(h, a, hg, ag, days, xi=0.004)

    flat_a = flat.attack[list(flat.teams).index("A")]
    decayed_a = decayed.attack[list(decayed.teams).index("A")]

    # With no decay the two blocks cancel and A rates as average; with decay the
    # recent block dominates and A rates well above average.
    assert flat_a == pytest.approx(0.0, abs=0.05)
    assert decayed_a > 0.3


def test_stale_teams_are_dropped_from_the_fit():
    """A team whose matches are all ancient carries no usable information."""
    h, a, hg, ag, days, *_ = _simulate(n_teams=8, n_rounds=40)
    # Bolt on a team that only played 12 years ago.
    h = np.concatenate([h, ["Ghost", "T00"]])
    a = np.concatenate([a, ["T00", "Ghost"]])
    hg = np.concatenate([hg, [1.0, 1.0]])
    ag = np.concatenate([ag, [1.0, 1.0]])
    days = np.concatenate([days, [4400.0, 4400.0]])

    fit = fit_dixon_coles(h, a, hg, ag, days, xi=0.0018)
    assert "Ghost" in fit.dropped_teams
    assert "Ghost" not in fit.teams
    assert not fit.knows("Ghost")


def test_knows_respects_the_minimum_match_count():
    h, a, hg, ag, days, *_ = _simulate()
    fit = fit_dixon_coles(h, a, hg, ag, days, xi=0.0)
    team = fit.teams[0]
    assert fit.knows(team, min_matches=1)
    assert not fit.knows(team, min_matches=10_000)


def test_trim_keeps_everything_when_all_teams_are_current():
    h = np.array(["A", "B", "C", "A"])
    a = np.array(["B", "C", "A", "C"])
    w = np.ones(4)
    assert _trim_stale_teams(h, a, w, min_effective=1.0).all()


def test_fit_needs_at_least_two_teams():
    with pytest.raises(ValueError):
        fit_dixon_coles(
            np.array(["A"]), np.array(["A"]), np.array([1.0]), np.array([0.0]), np.array([1.0])
        )
