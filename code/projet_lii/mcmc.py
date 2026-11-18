"""MCMC (emcee) sur theta = (ln d_pg, ln sigma_g, ln A), prior uniforme en ln sur une boîte.
Même modèle, même bruit et mêmes données que grid_posterior.py (rng(0), N=100 puis N=1).

Validation en deux temps :
  1. A fixé à sa vraie valeur -> doit reproduire la grille 2D de grid_posterior.py ;
  2. A inconnu (3 paramètres) -> comparé à une grille 3D (d_pg, sigma_g, A) marginalisée sur A.
Diagnostics : temps d'autocorrélation intégré (emcee), R-hat rang-normalisé « split »
(Vehtari et al. 2021) entre ensembles indépendants, ESS = n_échantillons / tau.
"""
import numpy as np
import emcee
import lognormal as L
import grid_posterior as G

A_TRUE = 1000.0
# Boîte du prior (uniforme en ln). d_grid de la bibliothèque : 2-200 nm.
BOX = dict(dpg=(5e-9, 60e-9), sg=(1.01, 1.9), A=(1e2, 1e4))
LO = np.log([BOX["dpg"][0], BOX["sg"][0], BOX["A"][0]])
HI = np.log([BOX["dpg"][1], BOX["sg"][1], BOX["A"][1]])


class Model:
    """Signal polydisperse vectorisé : s(dpg, sg) pour un lot de couples en un produit matriciel."""

    def __init__(self, t, d_grid):
        self.t, self.d = t, d_grid
        _, self.S = L.library(t, d_grid)                     # (n_d, n_t)
        # poids de la règle des trapèzes sur la grille non uniforme (= np.trapezoid)
        dd = np.diff(d_grid)
        self.q = np.zeros_like(d_grid)
        self.q[:-1] += dd / 2; self.q[1:] += dd / 2

    def shape(self, dpg, sg):
        """Signal normalisé au pic, dpg et sg de forme (n,) -> (n, n_t)."""
        dpg, sg = np.atleast_1d(dpg), np.atleast_1d(sg)
        w = L.lognormal_pdf(self.d[None, :], dpg[:, None], sg[:, None]) * self.q
        s = w @ self.S
        return s / s.max(axis=1, keepdims=True)


def loglike_vec(b, s_mod, gam, the, tau, N):
    """Même vraisemblance que grid_posterior.loglike, pour s_mod de forme (n, n_t)."""
    v = G.variance(s_mod, gam, the, tau, N)
    return -0.5 * np.sum((b - s_mod) ** 2 / v + np.log(v), axis=-1)


def make_log_prob(model, b, nz, A_fixed=None):
    def log_prob(th):                                       # th : (n_walkers, ndim)
        th = np.atleast_2d(th)
        full = th if A_fixed is None else np.column_stack([th, np.full(len(th), np.log(A_fixed))])
        ok = np.all((full > LO) & (full < HI), axis=1)
        lp = np.full(len(th), -np.inf)
        if ok.any():
            f = full[ok]
            s = np.exp(f[:, 2:3]) * model.shape(np.exp(f[:, 0]), np.exp(f[:, 1]))
            lp[ok] = loglike_vec(b, s, **nz)
        return lp
    return log_prob


# ---------------------------------------------------------------- diagnostics
def rank_normalized_split_rhat(chains):
    """chains : (n_chains, n_draws). R-hat rang-normalisé split (Vehtari 2021), sans le 'folded'."""
    from scipy.stats import norm, rankdata
    m, n = chains.shape
    half = n // 2
    x = np.concatenate([chains[:, :half], chains[:, half:2 * half]])   # 2m demi-chaînes
    r = rankdata(x, axis=None).reshape(x.shape)
    z = norm.ppf((r - 0.375) / (x.size + 0.25))
    mm, nn = z.shape
    B = nn * z.mean(axis=1).var(ddof=1)
    W = z.var(axis=1, ddof=1).mean()
    return np.sqrt(((nn - 1) / nn * W + B / nn) / W)


def find_start(log_prob, rng, ndim):
    """Point tiré uniformément dans la boîte -> Nelder-Mead sur -log p (2 relances).
    Un départ directement uniforme dans la boîte laisse des marcheurs bloqués aux bords :
    la postérieure est trop piquée pour que le « stretch move » les ramène."""
    from scipy.optimize import minimize
    x = LO[:ndim] + rng.random(ndim) * (HI - LO)[:ndim]
    f = lambda th: -log_prob(th[None])[0] if np.isfinite(log_prob(th[None])[0]) else 1e300
    for _ in range(3):
        x = minimize(f, x, method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-4, maxiter=4000)).x
    return x


def run(model, b, nz, A_fixed=None, n_ens=4, nwalkers=32, nsteps=6000, seed=1):
    """n_ens ensembles indépendants, chacun démarré d'une optimisation partant d'un point
    aléatoire de la boîte ; marcheurs dans une petite boule autour du point trouvé."""
    ndim = 3 if A_fixed is None else 2
    log_prob = make_log_prob(model, b, nz, A_fixed)
    rng = np.random.default_rng(seed)
    out, starts = [], []
    for k in range(n_ens):
        x0 = find_start(log_prob, rng, ndim); starts.append(np.exp(x0))
        p0 = x0 + 1e-4 * rng.standard_normal((nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True)
        sampler.random_state = np.random.RandomState(int(rng.integers(2**31))).get_state()
        sampler.run_mcmc(p0, nsteps, progress=False)
        out.append(sampler)
    # burn-in : 5 tau (la boule initiale doit se dilater jusqu'à la taille de la postérieure)
    taus = np.array([emcee.autocorr.integrated_time(s.get_chain()[nsteps // 2:], quiet=True)
                     for s in out])
    burn = int(5 * taus.max())
    if burn > nsteps // 2:
        print(f"  ATTENTION : burn-in {burn} > nsteps/2 ; allonger les chaînes")
    chains = np.array([s.get_chain(discard=burn) for s in out])        # (n_ens, n, nw, ndim)
    taus = np.array([emcee.autocorr.integrated_time(c, quiet=True) for c in chains])
    # R-hat : chaque marcheur de chaque ensemble est une chaîne
    flat_chains = chains.transpose(0, 2, 1, 3).reshape(-1, chains.shape[1], ndim)
    rhat = np.array([rank_normalized_split_rhat(flat_chains[..., j]) for j in range(ndim)])
    n_samp = chains.shape[1] * nwalkers * n_ens
    ess = n_samp / taus.mean(axis=0)
    acc = np.mean([s.acceptance_fraction.mean() for s in out])
    samples = chains.reshape(-1, ndim)
    if A_fixed is not None:
        samples = np.column_stack([samples, np.full(len(samples), np.log(A_fixed))])
    min_acc = min(s.acceptance_fraction.min() for s in out)
    return dict(samples=np.exp(samples), tau=taus, rhat=rhat, ess=ess, burn=burn,
                nsteps=nsteps, acc=acc, min_acc=min_acc, starts=np.array(starts))


# ---------------------------------------------------------------- grille de référence
def grid_reference(model, b, nz, samples, A_fixed=None, n=90, nA=60):
    """Grille sur ±6 écarts-types des échantillons MCMC. Si A libre : grille 3D, somme sur A."""
    def ax(x, nn, lo, hi):
        m, s = x.mean(), x.std()
        return np.linspace(max(m - 6 * s, lo), min(m + 6 * s, hi), nn)
    dpg_ax = ax(samples[:, 0], n, *BOX["dpg"])
    sg_ax = ax(samples[:, 1], n, *BOX["sg"])
    D, Gs = np.meshgrid(dpg_ax, sg_ax)                    # (n_sg, n_dpg)
    shp = model.shape(D.ravel(), Gs.ravel())              # (n*n, n_t)
    if A_fixed is not None:
        lp = loglike_vec(b, A_fixed * shp, **nz)
        A_ax, lp3 = np.array([A_fixed]), lp[:, None]
    else:
        A_ax = ax(samples[:, 2], nA, *BOX["A"])
        lp3 = np.empty((len(shp), nA))
        for j, A in enumerate(A_ax):
            lp3[:, j] = loglike_vec(b, A * shp, **nz)
    # prior uniforme en ln -> densité en variables physiques ∝ 1/(dpg sg A)
    lp3 = lp3 - np.log(D.ravel() * Gs.ravel())[:, None] - np.log(A_ax)[None, :]
    p3 = np.exp(lp3 - lp3.max()); p3 /= p3.sum()
    post2 = p3.sum(axis=1).reshape(D.shape)
    return dict(dpg=dpg_ax, sg=sg_ax, A=A_ax, D=D, G=Gs, post=post2,
                mA=p3.sum(axis=0))


def summarize(x):
    return x.mean(), x.std()


def grid_moments(g):
    m_d, m_s = g["post"].sum(0), g["post"].sum(1)
    mu_d = (m_d * g["dpg"]).sum(); sd_d = np.sqrt((m_d * (g["dpg"] - mu_d) ** 2).sum())
    mu_s = (m_s * g["sg"]).sum(); sd_s = np.sqrt((m_s * (g["sg"] - mu_s) ** 2).sum())
    d32 = L.sauter(g["D"], g["G"]); mu_32 = (g["post"] * d32).sum()
    sd_32 = np.sqrt((g["post"] * (d32 - mu_32) ** 2).sum())
    mu_A = (g["mA"] * g["A"]).sum(); sd_A = np.sqrt((g["mA"] * (g["A"] - mu_A) ** 2).sum())
    return dict(dpg=(mu_d, sd_d), sg=(mu_s, sd_s), d32=(mu_32, sd_32), A=(mu_A, sd_A))


if __name__ == "__main__":
    import time
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, 2e-6, 2001)
    d_grid = np.geomspace(2e-9, 200e-9, 120)
    model = Model(t, d_grid)

    # contrôle : forme vectorisée == lognormal.poly_signal
    e = abs(model.shape(20e-9, 1.3)[0] - L.poly_signal(20e-9, 1.3, d_grid, model.S)).max()
    print(f"écart forme vectorisée / poly_signal : {e:.1e}")

    dpg0, sg0 = 20e-9, 1.3
    s_true = A_TRUE * L.poly_signal(dpg0, sg0, d_grid, model.S)
    d32_0 = L.sauter(dpg0, sg0)
    rng = np.random.default_rng(0)                # mêmes données que grid_posterior.py
    data = {N: G.make_data(s_true, rng, **dict(G.NOISE, N=N)) for N in (100, 1)}

    C_MC, C_GR = "#2f6fb3", "#333333"
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for row, N in enumerate((100, 1)):
        nz = dict(G.NOISE, N=N)
        b = data[N]
        for label, A_fixed in (("A connue", A_TRUE), ("A inconnue", None)):
            t0 = time.time()
            r = run(model, b, nz, A_fixed=A_fixed)
            g = grid_reference(model, b, nz, r["samples"], A_fixed=A_fixed)
            gm = grid_moments(g)
            smp = r["samples"]
            d32 = L.sauter(smp[:, 0], smp[:, 1])
            print(f"\nN={N:3d}, {label}  ({time.time() - t0:.0f} s, {r['nsteps']} pas, "
                  f"burn {r['burn']}, acceptation moy {r['acc']:.2f} / min {r['min_acc']:.2f})")
            print("  départs (optim.) d_pg nm / sg : " + " | ".join(
                f"{s[0] * 1e9:.2f}/{s[1]:.3f}" for s in r["starts"]))
            names = ["ln d_pg", "ln sg", "ln A"][:len(r["rhat"])]
            print("  tau (par ensemble) : " + " | ".join(
                f"{nm} " + "/".join(f"{x:.0f}" for x in r["tau"][:, j]) for j, nm in enumerate(names)))
            print("  R-hat : " + "  ".join(f"{nm} {x:.4f}" for nm, x in zip(names, r["rhat"]))
                  + "   ESS : " + "  ".join(f"{nm} {x:.0f}" for nm, x in zip(names, r["ess"])))
            rows = [("d_pg (nm)", smp[:, 0] * 1e9, gm["dpg"], 1e9, dpg0 * 1e9),
                    ("sigma_g", smp[:, 1], gm["sg"], 1, sg0),
                    ("d32 (nm)", d32 * 1e9, gm["d32"], 1e9, d32_0 * 1e9)]
            if A_fixed is None:
                rows.append(("A", smp[:, 2], gm["A"], 1, A_TRUE))
            corr = np.corrcoef(smp[:, 0], smp[:, 1])[0, 1]
            for nm, x, (gmu, gsd), sc, tru in rows:
                print(f"  {nm:9s} MCMC {x.mean():8.3f} ± {x.std():.3f} | grille "
                      f"{gmu * sc:8.3f} ± {gsd * sc:.3f} | vrai {tru:.3f}")
            print(f"  corr(d_pg, sg) MCMC {corr:+.3f}")
            if A_fixed is not None:
                continue

            # ---------- figure (A inconnue) : marginales MCMC vs grille 3D
            ax = axes[row]
            for a, (nm, x, sc_ax, sc, tru, marg) in zip(ax[:3], (
                    ("d_pg (nm)", smp[:, 0] * 1e9, g["dpg"], 1e9, dpg0 * 1e9, g["post"].sum(0)),
                    ("σ_g", smp[:, 1], g["sg"], 1, sg0, g["post"].sum(1)),
                    ("A (comptes)", smp[:, 2], g["A"], 1, A_TRUE, g["mA"]))):
                a.hist(x, bins=60, density=True, color=C_MC, alpha=0.55, label="MCMC (emcee)")
                xs = sc_ax * sc
                a.plot(xs, marg / np.trapezoid(marg, xs), color=C_GR, lw=1.5,
                       label="grille 3D")
                a.axvline(tru, color=C_GR, ls=":", lw=1, label="valeur vraie")
                a.set(xlabel=nm, yticks=[], title=f"N={N} tir(s) — {nm}")
                a.spines[["top", "right", "left"]].set_visible(False)
            ax[0].legend(fontsize=8, frameon=False)
            a = ax[3]
            sub = smp[np.random.default_rng(2).choice(len(smp), 4000, replace=False)]
            a.plot(sub[:, 0] * 1e9, sub[:, 1], ".", ms=1.5, color=C_MC, alpha=0.4,
                   label="échantillons MCMC")
            a.contour(g["D"] * 1e9, g["G"], g["post"], colors=C_GR, linewidths=1,
                      levels=sorted([G.hpd_level(g["post"], .95), G.hpd_level(g["post"], .68)]))
            a.plot(dpg0 * 1e9, sg0, "*", color="#c0392b", ms=11, label="valeur vraie")
            a.set(xlabel="d_pg (nm)", ylabel="σ_g", title=f"N={N} — grille 68/95 % (traits)")
            a.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("A posteriori avec amplitude A inconnue : MCMC vs grille 3D marginalisée sur A")
    fig.tight_layout(); fig.savefig("mcmc_posterior.png", dpi=120)
    print("\n-> mcmc_posterior.png")
