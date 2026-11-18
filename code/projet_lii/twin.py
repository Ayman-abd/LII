"""Expériences jumelles : couverture des régions de crédibilité 68 % / 95 %.

Pour chaque couple vrai (d_pg, sigma_g) et chaque modèle de génération du bruit, on tire
n_rep réalisations de données, on calcule la postérieure (A inconnue, prior uniforme en ln)
et on compte combien de fois la valeur vraie tombe dans :
  - la région HPD 2D de (ln d_pg, ln sigma_g), marginalisée sur A ;
  - les intervalles centraux 1D de d_pg, sigma_g, d32 et A.

Inversion = grille « exacte » (pas de MCMC, trop lent pour des milliers d'inversions) :
  * MAP (Nelder-Mead) + hessien -> covariance de Laplace ;
  * grille n x n dans le plan (ln d_pg, ln sigma_g) BLANCHI (suit la vallée d32, corr -0,99),
    sur ±k écarts-types ; 2e passe recentrée sur la moyenne/covariance de la 1re grille ;
  * A intégrée pour chaque forme par trapèzes en ln A sur ±6 écarts-types conditionnels autour
    de l'estimateur des moindres carrés pondérés.
Validée contre mcmc.py (python twin.py --check).

Génération du bruit :
  'indep'  : Var = (gam^2 + the s + tau^2 s^2)/N, indépendant à chaque instant (= modèle d'inversion)
  'common' : le terme tau multiplie toute la courbe : b = s (1 + tau xi/sqrt(N)) + bruit(gam, the)
             -> l'inversion garde le modèle 'indep' (mal spécifié).
"""
import argparse
import time
import numpy as np
from scipy.optimize import minimize
import lognormal as L
import grid_posterior as G
import mcmc as M

T_AX = np.linspace(0, 2e-6, 2001)
D_GRID = np.geomspace(2e-9, 200e-9, 120)
TRUTHS = [(20e-9, 1.3), (12e-9, 1.6), (30e-9, 1.15)]
A_TRUE = 1000.0
LEVELS = (0.68, 0.95)


def make_data(s_true, rng, gam, the, tau, N, mode="indep"):
    if mode == "indep":
        return G.make_data(s_true, rng, gam=gam, the=the, tau=tau, N=N)
    xi = rng.standard_normal()
    return (s_true * (1 + tau * xi / np.sqrt(N))
            + rng.standard_normal(s_true.size) * np.sqrt((gam**2 + the * s_true) / N))


def loglike_fast(b, s, gam, the, tau, N):
    """= mcmc.loglike_vec (écart < 1e-10), avec moins de tableaux temporaires (~1,6x)."""
    v = s * (tau * tau); v += the; v *= s; v += gam * gam
    r = b - s; r *= r; r /= v; r *= N
    np.log(v, out=v)
    return -0.5 * (r.sum(-1) + v.sum(-1) - s.shape[-1] * np.log(N))


class Inverter:
    def __init__(self, model, nz, n=(30, 44), k=6.0, nA=11):
        self.m, self.nz, self.n, self.k, self.nA = model, nz, n, k, nA

    # --- A intégrée numériquement pour un lot de formes ------------------------------------
    def _marg_A(self, b, shapes, X):
        """shapes (n_s, n_t), X (n_s, 2) en ln. Renvoie log m(X) (densité marginale en ln,
        à une constante près commune), et les noeuds / log-densités en ln A."""
        # centre : moindres carrés repondérés sur le modèle (IRLS, 3 itérations) ;
        # des poids tirés des données bruitées biaisent A (vu en N=1)
        v = G.variance(np.maximum(b, 1.0), **self.nz)[None, :]
        for _ in range(3):
            ss = (shapes**2 / v).sum(1)
            A_hat = np.clip((shapes * b / v).sum(1) / ss, 1.0, None)
            v = G.variance(A_hat[:, None] * shapes, **self.nz)
        # 1er passage large (±12 sd « moindres carrés »), puis recentrage sur les moments
        # conditionnels en ln A et 2e passage sur ±6 sd. Le 2e passage est indispensable quand
        # le modèle de bruit est faux (τ commun) : la largeur réelle dépasse alors celle des MC.
        mu = np.log(A_hat); sd = 1.0 / np.sqrt(ss) / A_hat
        for umax in (12.0, 6.0):
            u = np.linspace(-umax, umax, self.nA)
            lnA = mu[:, None] + u[None, :] * sd[:, None]                  # (n_s, nA)
            lp = np.empty_like(lnA)
            for j in range(self.nA):
                lp[:, j] = loglike_fast(b, np.exp(lnA[:, j:j + 1]) * shapes, **self.nz)
            th = np.column_stack([np.repeat(X, self.nA, 0), lnA.ravel()])
            inside = np.all((th > M.LO) & (th < M.HI), axis=1).reshape(lp.shape)
            lp = np.where(inside, lp, -np.inf)
            c = lp.max(1, keepdims=True); c = np.where(np.isfinite(c), c, 0.0)
            p = np.exp(lp - c); P = p.sum(1)
            ok = P > 0
            mu = np.where(ok, (p * lnA).sum(1) / np.where(ok, P, 1), mu)
            sd = np.where(ok, np.sqrt((p * (lnA - mu[:, None])**2).sum(1) / np.where(ok, P, 1)), sd)
            sd = np.maximum(sd, 1e-6)
        h = lnA[:, 1] - lnA[:, 0]                                         # pas en ln A
        with np.errstate(divide="ignore"):
            logm = np.log(P * h) + c[:, 0]
        return logm, lnA, lp

    def _grid(self, b, mu, C, n):
        Lc = np.linalg.cholesky(C)
        u = np.linspace(-self.k, self.k, n)
        U = np.array(np.meshgrid(u, u)).reshape(2, -1)
        X = (mu[:, None] + Lc @ U).T                                      # (n², 2)
        # noeuds hors de la boîte du prior : densité nulle (et sg < 1 n'a pas de sens)
        ins = np.all((X > M.LO[:2]) & (X < M.HI[:2]), axis=1)
        logm = np.full(len(X), -np.inf)
        lnA = np.zeros((len(X), self.nA)); lpA = np.full((len(X), self.nA), -np.inf)
        shapes = self.m.shape(np.exp(X[ins, 0]), np.exp(X[ins, 1]))
        logm[ins], lnA[ins], lpA[ins] = self._marg_A(b, shapes, X[ins])
        w = np.exp(logm - logm.max()); w /= w.sum()
        g2 = w.reshape(n, n)
        edge = g2[[0, -1], :].sum() + g2[:, [0, -1]].sum()               # bord de la grille 2D
        with np.errstate(invalid="ignore"):
            pA = np.exp(lpA - lpA.max(1, keepdims=True))
            fA = np.nan_to_num(pA[:, [0, -1]].sum(1) / pA.sum(1))
        edgeA = (fA * w).sum()                                            # bord de la grille en A
        return dict(X=X, logm=logm, w=w, lnA=lnA, lpA=lpA, mx=logm.max(),
                    edge=edge, edgeA=edgeA)

    def invert(self, b):
        lp = M.make_log_prob(self.m, b, self.nz)
        f = lambda th: -lp(th[None])[0] if np.isfinite(lp(th[None])[0]) else 1e300
        # départ générique (n'utilise pas la valeur vraie) ; A par moindres carrés
        s0 = self.m.shape(15e-9, 1.4)[0]
        x = np.log([15e-9, 1.4, max((s0 * b).sum() / (s0 * s0).sum(), 10.0)])
        for _ in range(3):
            x = minimize(f, x, method="Nelder-Mead",
                         options=dict(xatol=1e-8, fatol=1e-6, maxiter=5000)).x
        # hessien par différences finies -> covariance de Laplace du bloc (d_pg, sg)
        hh = 2e-4; H = np.empty((3, 3)); E = np.eye(3) * hh
        for i in range(3):
            for j in range(i, 3):
                H[i, j] = H[j, i] = -(lp((x + E[i] + E[j])[None])[0] - lp((x + E[i] - E[j])[None])[0]
                                      - lp((x - E[i] + E[j])[None])[0] + lp((x - E[i] - E[j])[None])[0]) / (4 * hh * hh)
        # MAP sur la borne du prior (sg -> 1.01) : hessien non fini -> covariance de repli
        try:
            if not np.all(np.isfinite(H)):
                raise np.linalg.LinAlgError
            C = np.linalg.inv(H)[:2, :2]; np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = np.diag([0.1**2, 0.05**2])
        g = self._grid(b, x[:2], C, self.n[0])
        # 2e passe, plus fine : recentrée sur les moments de la 1re (corrige la non-gaussianité)
        mu = g["w"] @ g["X"]; d = g["X"] - mu
        g = self._grid(b, mu, (g["w"][:, None] * d).T @ d, self.n[1])
        g["map"] = x
        return g

    def density_at(self, b, g, dpg, sg):
        """log m au point vrai, sur la même échelle que g['logm']."""
        X = np.log([[dpg, sg]])
        logm, _, _ = self._marg_A(b, self.m.shape(dpg, sg), X)
        return logm[0]


def wquantile(x, w, q):
    i = np.argsort(x); cw = np.cumsum(w[i]); cw /= cw[-1]
    return np.interp(q, cw, x[i])


def summarize(g):
    """Échantillons pondérés des quantités d'intérêt (noeuds de la grille)."""
    dpg, sg = np.exp(g["X"][:, 0]), np.exp(g["X"][:, 1])
    # masse jointe d'un noeud (forme, A) ∝ densité × pas en ln A (pas propre à chaque forme)
    pA = np.exp(g["lpA"] - g["lpA"][np.isfinite(g["lpA"])].max()) \
        * (g["lnA"][:, 1] - g["lnA"][:, 0])[:, None]
    wA = pA / pA.sum()
    return dict(dpg=(dpg, g["w"]), sg=(sg, g["w"]), d32=(L.sauter(dpg, sg), g["w"]),
                A=(np.exp(g["lnA"]).ravel(), wA.ravel()))


def one_case(args):
    """Toutes les réalisations d'un cas. Exécuté dans un processus fils."""
    (dpg0, sg0), N, mode, n_rep, seed = args
    model = _MODEL
    nz = dict(G.NOISE, N=N)
    inv = Inverter(model, nz)
    s_true = A_TRUE * L.poly_signal(dpg0, sg0, D_GRID, model.S)
    truth = dict(dpg=dpg0, sg=sg0, d32=L.sauter(dpg0, sg0), A=A_TRUE)
    rng = np.random.default_rng(seed)
    rec = {key: [] for key in ("hpd", "dpg", "sg", "d32", "A", "mean", "sd", "edge")}
    for _ in range(n_rep):
        b = make_data(s_true, rng, **nz, mode=mode)
        g = inv.invert(b)
        # HPD 2D : vrai dans la région <=> densité(vrai) >= seuil
        w = np.sort(g["w"])[::-1]; cw = np.cumsum(w)
        lvl = [w[np.searchsorted(cw, q)] for q in LEVELS]
        p_true = np.exp(inv.density_at(b, g, dpg0, sg0) - g["mx"]) / np.exp(g["logm"] - g["mx"]).sum()
        rec["hpd"].append([p_true >= l for l in lvl])
        sm = summarize(g)
        mean, sd = {}, {}
        for key, (x, wx) in sm.items():
            iv = [(wquantile(x, wx, (1 - q) / 2), wquantile(x, wx, (1 + q) / 2)) for q in LEVELS]
            rec[key].append([lo <= truth[key] <= hi for lo, hi in iv])
            mean[key] = (wx * x).sum() / wx.sum()
            sd[key] = np.sqrt((wx * (x - mean[key])**2).sum() / wx.sum())
        rec["mean"].append([mean[k] for k in ("dpg", "sg", "d32", "A")])
        rec["sd"].append([sd[k] for k in ("dpg", "sg", "d32", "A")])
        rec["edge"].append(max(g["edge"], g["edgeA"]))
    return {k: np.array(v) for k, v in rec.items()}


_MODEL = None
def _init():
    global _MODEL
    _MODEL = M.Model(T_AX, D_GRID)


def check():
    """Inverseur grille vs MCMC sur les données N=1 de mcmc.py (A inconnue)."""
    _init()
    s_true = A_TRUE * L.poly_signal(20e-9, 1.3, D_GRID, _MODEL.S)
    rng = np.random.default_rng(0)
    data = {N: G.make_data(s_true, rng, **dict(G.NOISE, N=N)) for N in (100, 1)}
    ref = {100: dict(dpg=(19.995, .146), sg=(1.300, .004), d32=(23.747, .076), A=(999.665, 2.237)),
           1: dict(dpg=(19.025, 1.395), sg=(1.327, .040), d32=(23.235, .743), A=(1002.453, 22.160))}
    for N in (100, 1):
        inv = Inverter(_MODEL, dict(G.NOISE, N=N))
        t0 = time.time(); g = inv.invert(data[N]); dt = time.time() - t0
        print(f"N={N}: inversion {dt:.2f} s, masse au bord : grille 2D {g['edge']:.1e}, A {g['edgeA']:.1e}")
        for key, (x, w) in summarize(g).items():
            sc = 1e9 if key in ("dpg", "d32") else 1
            mu = (w * x).sum() / w.sum(); sd = np.sqrt((w * (x - mu)**2).sum() / w.sum())
            print(f"  {key:4s} grille {mu * sc:9.3f} ± {sd * sc:7.3f} | MCMC {ref[N][key][0]:9.3f} ± {ref[N][key][1]:7.3f}")


def report(out, n):
    """Tableau de couverture ; renvoie les lignes pour la figure."""
    tol = {q: 2 * np.sqrt(q * (1 - q) / n) * 100 for q in LEVELS}
    print(f"\nCouverture (%) sur {n} réalisations ; tolérance ±2σ binomiale : "
          f"68 % ±{tol[.68]:.1f}, 95 % ±{tol[.95]:.1f}   (* = hors tolérance)")
    print(f"{'N':>4} {'bruit':7} {'d_pg/sg vrai':>13} | " +
          " | ".join(f"{k:>11}" for k in ("HPD 2D", "d_pg", "σ_g", "d32", "A")) +
          " | z(d32) moy/σ | bord max")
    rows = []
    for (tr, N, mode), r in out.items():
        cov = {k: r[k].mean(0) * 100 for k in ("hpd", "dpg", "sg", "d32", "A")}
        z = (r["mean"][:, 2] - L.sauter(*tr)) / r["sd"][:, 2]
        flag = lambda c, q: "*" if abs(c - q * 100) > tol[q] else " "
        print(f"{N:4d} {mode:7} {tr[0] * 1e9:5.0f}nm/{tr[1]:.2f} | " + " | ".join(
            f"{c[0]:4.1f}{flag(c[0], .68)}/{c[1]:4.1f}{flag(c[1], .95)}" for c in cov.values())
            + f" | {z.mean():+.2f}/{z.std():.2f} | {r['edge'].max():.0e}")
        rows.append((tr, N, mode, cov))
    return rows, tol


def plot(rows, tol, n, Ns, fname="twin_coverage.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("hpd", "HPD 2D"), ("dpg", "d_pg"), ("sg", "σ_g"), ("d32", "d32"), ("A", "A")]
    colors = {"indep": "#2f6fb3", "common": "#d9822b"}
    fig, axs = plt.subplots(len(Ns), 2, figsize=(13, 3.8 * len(Ns)), squeeze=False)
    for ri, N in enumerate(Ns):
        for ci, q in enumerate(LEVELS):
            ax = axs[ri, ci]
            ax.axhspan(q * 100 - tol[q], q * 100 + tol[q], color="0.9", zorder=0)
            ax.axhline(q * 100, color="0.3", lw=1, zorder=1)
            for ti, tr in enumerate(TRUTHS):
                for ki, (k, _) in enumerate(keys):
                    for mode, dx in (("indep", -0.15), ("common", 0.15)):
                        c = [cv for (t_, n_, m_, cv) in rows if t_ == tr and n_ == N and m_ == mode][0]
                        ax.plot(ki * 4 + ti + dx, c[k][ci], "o", ms=6, color=colors[mode],
                                label=(f"τ {'indépendant' if mode == 'indep' else 'commun'} à la génération"
                                       if ti == 0 and ki == 0 else None))
            ax.set_xticks([ki * 4 + 1 for ki in range(len(keys))], [lab for _, lab in keys])
            for ki in range(1, len(keys)):
                ax.axvline(ki * 4 - 1, color="0.85", lw=0.8)
            ax.set(ylabel="couverture empirique (%)", ylim=(0, 102),
                   title=f"N={N} tir(s) — niveau nominal {q * 100:.0f} %")
            ax.spines[["top", "right"]].set_visible(False)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"Expériences jumelles, A inconnue — {n} réalisations par cas, bande grise = ±2σ binomiale\n"
                 "dans chaque groupe, de gauche à droite : " +
                 ", ".join(f"{t[0] * 1e9:.0f} nm / σ_g {t[1]}" for t in TRUTHS), fontsize=10)
    fig.tight_layout(); fig.savefig(fname, dpi=120)
    print(f"-> {fname}")


if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--plot", action="store_true", help="refait tableau + figure depuis twin_results.npy")
    ap.add_argument("--n_rep", type=int, default=300)
    ap.add_argument("--N", type=int, nargs="+", default=[1, 100])
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.check:
        check(); raise SystemExit
    if a.plot:
        d = np.load("twin_results.npy", allow_pickle=True).item()
        Ns = sorted({N for (_, N, _) in d["out"]})
        plot(*report(d["out"], d["n_rep"]), d["n_rep"], Ns); raise SystemExit

    # 1 thread BLAS par processus : sinon les processus se disputent les cœurs (campagne
    # de nov. 2026 : 2 h 34 au lieu de ~1 h). Les fils importent numpy après ce réglage.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    from multiprocessing import Pool
    cases = [(tr, N, mode) for N in a.N for mode in ("indep", "common") for tr in TRUTHS]
    chunk = 25                              # paquets : occupation des processus + reprise
    jobs = [(tr, N, mode, chunk, 1000 * i + j)
            for i, (tr, N, mode) in enumerate(cases) for j in range(a.n_rep // chunk)]
    os.makedirs("twin_parts", exist_ok=True)
    part = lambda job: f"twin_parts/{job[2]}_N{job[1]}_{job[0][0] * 1e9:.0f}_{job[0][1]}_{job[4]}.npy"
    todo = [j for j in jobs if not os.path.exists(part(j))]
    print(f"{len(jobs)} paquets de {chunk}, {len(jobs) - len(todo)} déjà faits (reprise)", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for k, (job, r) in enumerate(zip(todo, pool.imap(one_case, todo, chunksize=1))):
            np.save(part(job), r, allow_pickle=True)
            el = time.time() - t0
            print(f"  {k + 1}/{len(todo)} paquets, {el / 60:.1f} min, reste ~{el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min",
                  flush=True)
    print(f"{len(todo) * chunk} inversions en {time.time() - t0:.0f} s")

    out = {}
    for i, case in enumerate(cases):
        parts = [np.load(part(j), allow_pickle=True).item() for j in jobs if j[4] // 1000 == i]
        out[case] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.save("twin_results.npy", dict(out=out, n_rep=a.n_rep), allow_pickle=True)
    plot(*report(out, a.n_rep), a.n_rep, a.N)
