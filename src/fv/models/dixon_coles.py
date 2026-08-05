"""Time-decayed Dixon-Coles bivariate Poisson.

Each team gets an attack rating and a defence rating; each league gets a home
advantage. Goal expectations are

    lambda_home = exp(attack_home + defence_away + home_adv)
    lambda_away = exp(attack_away + defence_home)

Independent Poisson alone misprices low-scoring games — real 0-0 and 1-1 results are
more common than independence implies — so Dixon and Coles (1997) add a correction
``tau`` on the four scorelines where both teams score at most one.

Matches are weighted by ``exp(-xi * days_before_cutoff)``, so recent form counts for
more. ``xi`` is tuned on validation seasons rather than guessed: see :func:`tune_xi`.

Identifiability: shifting every attack rating up by c and every defence rating down
by c leaves both goal expectations unchanged, so the likelihood has one flat
direction. It is removed by constraining the attack ratings to sum to zero, which is
done by construction (the last team's rating is minus the sum of the others) rather
than by a penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

RHO_BOUND = 0.4
TAU_FLOOR = 1e-12


def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction. 1.0 everywhere except the four cells."""
    t = np.ones_like(lam)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return np.clip(t, TAU_FLOOR, None)


@dataclass
class DixonColesFit:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    xi: float
    n_matches: dict[str, int] = field(default_factory=dict)
    converged: bool = True
    log_likelihood: float = 0.0
    effective_sample: float = 0.0
    dropped_teams: list[str] = field(default_factory=list)

    @property
    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def rates(self, home: str, away: str) -> tuple[float, float]:
        idx = self.index
        i, j = idx[home], idx[away]
        lam = float(np.exp(self.attack[i] + self.defence[j] + self.home_adv))
        mu = float(np.exp(self.attack[j] + self.defence[i]))
        return lam, mu

    def score_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        lam, mu = self.rates(home, away)
        h = poisson.pmf(np.arange(max_goals + 1), lam)
        a = poisson.pmf(np.arange(max_goals + 1), mu)
        m = np.outer(h, a)
        # Apply the low-score correction to the four affected cells.
        m[0, 0] *= 1.0 - lam * mu * self.rho
        m[0, 1] *= 1.0 + lam * self.rho
        m[1, 0] *= 1.0 + mu * self.rho
        m[1, 1] *= 1.0 - self.rho
        m = np.clip(m, 0.0, None)
        total = m.sum()
        return m / total if total > 0 else m

    def predict(self, home: str, away: str, max_goals: int = 10) -> tuple[float, float, float]:
        """Returns (p_home, p_draw, p_away), summing to 1."""
        m = self.score_matrix(home, away, max_goals)
        p_home = float(np.tril(m, -1).sum())
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def knows(self, team: str, min_matches: int = 0) -> bool:
        return team in self.index and self.n_matches.get(team, 0) >= min_matches


def _unpack(params: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    attack_free = params[: n - 1]
    attack = np.concatenate([attack_free, [-attack_free.sum()]])
    defence = params[n - 1 : 2 * n - 1]
    home_adv = params[2 * n - 1]
    rho = params[2 * n]
    return attack, defence, home_adv, rho


def _nll_and_gradient(
    params: np.ndarray,
    n: int,
    hi: np.ndarray,
    ai: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and its analytic gradient.

    The gradient is supplied explicitly because a numerical one needs an extra
    function evaluation per parameter, and with ~90 parameters per league that made
    a single fit take ten seconds and still stop at the iteration limit without
    converging. Differentiating by hand takes the same fit to well under a second.

    Working in terms of ``log lambda`` keeps it tidy, since every rating enters
    linearly there:

        d/d(log lam) [-lam + x*log lam] = x - lam
        d/d(log lam) [log tau]          = (d tau/d log lam) / tau
    """
    attack, defence, home_adv, rho = _unpack(params, n)
    log_lam = attack[hi] + defence[ai] + home_adv
    log_mu = attack[ai] + defence[hi]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    tau = np.ones_like(lam)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    tau = np.clip(tau, TAU_FLOOR, None)

    # Poisson log-pmf without the factorial term, which is constant in the params.
    ll = np.log(tau) + (-lam + hg * log_lam) + (-mu + ag * log_mu)
    nll = -float(np.sum(weights * ll))

    # d tau / d(log lam), d tau / d(log mu), d tau / d rho
    dt_dloglam = np.zeros_like(lam)
    dt_dlogmu = np.zeros_like(lam)
    dt_drho = np.zeros_like(lam)
    dt_dloglam[m00] = -lam[m00] * mu[m00] * rho
    dt_dlogmu[m00] = -lam[m00] * mu[m00] * rho
    dt_drho[m00] = -lam[m00] * mu[m00]
    dt_dloglam[m01] = lam[m01] * rho
    dt_drho[m01] = lam[m01]
    dt_dlogmu[m10] = mu[m10] * rho
    dt_drho[m10] = mu[m10]
    dt_drho[m11] = -1.0

    g_lam = weights * (dt_dloglam / tau + (hg - lam))
    g_mu = weights * (dt_dlogmu / tau + (ag - mu))
    g_rho = weights * (dt_drho / tau)

    # Attack enters via lam for the home side and via mu for the away side;
    # defence the other way round.
    d_attack = -(np.bincount(hi, g_lam, minlength=n) + np.bincount(ai, g_mu, minlength=n))
    d_defence = -(np.bincount(ai, g_lam, minlength=n) + np.bincount(hi, g_mu, minlength=n))
    d_home_adv = -float(g_lam.sum())
    d_rho = -float(g_rho.sum())

    # The last attack rating is minus the sum of the free ones, so its gradient
    # propagates back into every free parameter.
    grad = np.empty(2 * n + 1)
    grad[: n - 1] = d_attack[: n - 1] - d_attack[n - 1]
    grad[n - 1 : 2 * n - 1] = d_defence
    grad[2 * n - 1] = d_home_adv
    grad[2 * n] = d_rho

    return nll, grad


def _trim_stale_teams(
    home_teams: np.ndarray,
    away_teams: np.ndarray,
    weights: np.ndarray,
    min_effective: float,
    max_passes: int = 10,
) -> np.ndarray:
    """Mask out matches involving teams with negligible effective sample.

    Time decay means a club relegated fifteen years ago carries a weight around
    1e-7, so its attack and defence ratings are effectively unconstrained by the
    data. Left in the fit they wander freely, the optimiser never satisfies its
    convergence test, and every fit burns its full iteration budget for nothing.

    Dropping them is also the statistically honest move: we have no current opinion
    on a team that hasn't played in this division for a decade, and
    :meth:`DixonColesFit.knows` will report as much rather than inventing one.

    Removing a team lowers its opponents' effective weights too, so this iterates
    until the surviving set is stable.
    """
    keep = np.ones(len(home_teams), dtype=bool)
    for _ in range(max_passes):
        totals: dict[str, float] = {}
        for h, a, w in zip(home_teams[keep], away_teams[keep], weights[keep], strict=True):
            totals[h] = totals.get(h, 0.0) + w
            totals[a] = totals.get(a, 0.0) + w
        stale = {t for t, w in totals.items() if w < min_effective}
        if not stale:
            break
        new_keep = keep & ~(np.isin(home_teams, list(stale)) | np.isin(away_teams, list(stale)))
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return keep


def fit_dixon_coles(
    home_teams: np.ndarray,
    away_teams: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    days_before: np.ndarray,
    xi: float = 0.0018,
    init: DixonColesFit | None = None,
    maxiter: int = 500,
    min_effective_matches: float = 1.0,
) -> DixonColesFit:
    """Fit the model to a set of completed matches.

    ``days_before`` is each match's age in days relative to the training cutoff, and
    must be >= 0. ``init`` warm-starts from a previous fit, which matters in the
    walk-forward backtest: ratings barely move week to week, so warm-starting cuts
    the optimiser's work substantially.
    """
    home_teams = np.asarray(home_teams)
    away_teams = np.asarray(away_teams)
    weights_all = np.exp(-xi * np.asarray(days_before, dtype=float))

    keep = _trim_stale_teams(home_teams, away_teams, weights_all, min_effective_matches)
    dropped = sorted((set(home_teams) | set(away_teams)) - (set(home_teams[keep]) | set(away_teams[keep])))

    home_teams = home_teams[keep]
    away_teams = away_teams[keep]
    home_goals = np.asarray(home_goals)[keep]
    away_goals = np.asarray(away_goals)[keep]
    weights = weights_all[keep]

    teams = sorted(set(home_teams) | set(away_teams))
    n = len(teams)
    if n < 2:
        raise ValueError("need at least two teams with sufficient recent data to fit")

    idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[t] for t in home_teams], dtype=int)
    ai = np.array([idx[t] for t in away_teams], dtype=int)
    hg = np.asarray(home_goals, dtype=float)
    ag = np.asarray(away_goals, dtype=float)

    counts: dict[str, int] = {t: 0 for t in teams}
    for t in home_teams:
        counts[t] += 1
    for t in away_teams:
        counts[t] += 1

    x0 = np.zeros(2 * n + 1)
    x0[2 * n - 1] = 0.25  # home advantage starting point, roughly the historical value
    x0[2 * n] = -0.05  # rho is typically slightly negative
    if init is not None:
        prev = init.index
        for t, i in idx.items():
            if t in prev:
                if i < n - 1:
                    x0[i] = init.attack[prev[t]]
                x0[n - 1 + i] = init.defence[prev[t]]
        x0[2 * n - 1] = init.home_adv
        x0[2 * n] = init.rho

    bounds = [(-3.0, 3.0)] * (n - 1) + [(-3.0, 3.0)] * n + [(-1.0, 1.5)] + [
        (-RHO_BOUND, RHO_BOUND)
    ]

    res = minimize(
        _nll_and_gradient,
        x0,
        args=(n, hi, ai, hg, ag, weights),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8},
    )

    attack, defence, home_adv, rho = _unpack(res.x, n)
    return DixonColesFit(
        teams=teams,
        attack=attack,
        defence=defence,
        home_adv=float(home_adv),
        rho=float(rho),
        xi=xi,
        n_matches=counts,
        converged=bool(res.success),
        log_likelihood=-float(res.fun),
        effective_sample=float(weights.sum()),
        dropped_teams=dropped,
    )
