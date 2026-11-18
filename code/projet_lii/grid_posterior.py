"""Bruit de Sipkens 2017 + loi a posteriori sur grille 2D (d_pg, sigma_g).
Var(s) = gam^2 + the*s + tau^2*s^2, divisée par N tirs moyennés.
Valeurs de bruit PROVISOIRES (ordre de grandeur), à recaler.
"""
import numpy as np
import lognormal as L

NOISE = dict(gam=10.0, the=1.0, tau=0.2, N=100)   # signal en "comptes", pic = A
A = 1000.0                                          # amplitude au pic (supposée connue)

def variance(s, gam, the, tau, N):
    return (gam**2 + the*s + tau**2*s**2) / N

def make_data(s_true, rng, **nz):
    return s_true + rng.standard_normal(s_true.size) * np.sqrt(variance(s_true, **nz))

def loglike(b, s_mod, **nz):
    v = variance(s_mod, **nz)
    return -0.5*np.sum((b - s_mod)**2 / v + np.log(v))

def grid_posterior(b, d_grid, S, dpg_ax, sg_ax, **nz):
    """Prior uniforme sur la boîte -> posterior ∝ vraisemblance."""
    lp = np.array([[loglike(b, A*L.poly_signal(dpg, sg, d_grid, S), **nz)
                    for dpg in dpg_ax] for sg in sg_ax])          # (n_sg, n_dpg)
    post = np.exp(lp - lp.max())
    return post / post.sum()

def zoom(b, d_grid, S, dpg_ax, sg_ax, n=100, **nz):
    """Grille grossière puis resserrée autour de la zone où post > 1e-6 max."""
    for _ in range(3):
        post = grid_posterior(b, d_grid, S, dpg_ax, sg_ax, **nz)
        js, jd = np.where(post > 1e-6*post.max())
        dd, ds = dpg_ax[1]-dpg_ax[0], sg_ax[1]-sg_ax[0]
        dpg_ax = np.linspace(max(dpg_ax[jd.min()]-2*dd, 2e-9), dpg_ax[jd.max()]+2*dd, n)
        sg_ax = np.linspace(max(sg_ax[js.min()]-2*ds, 1.005), sg_ax[js.max()]+2*ds, n)
    return dpg_ax, sg_ax, grid_posterior(b, d_grid, S, dpg_ax, sg_ax, **nz)

def hpd_level(post, mass):
    """Seuil de densité tel que la région {post > seuil} contienne `mass`."""
    p = np.sort(post.ravel())[::-1]
    return p[np.searchsorted(np.cumsum(p), mass)]

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2e-6, 2001)
    d_grid = np.geomspace(2e-9, 200e-9, 120)
    _, S = L.library(t, d_grid)

    dpg0, sg0 = 20e-9, 1.3
    s_true = A*L.poly_signal(dpg0, sg0, d_grid, S)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    sg_line = np.linspace(1.005, 2.0, 300)
    for k, N in enumerate((100, 1)):
        nz = dict(NOISE, N=N)
        b = make_data(s_true, rng, **nz)
        dpg_ax, sg_ax, post = zoom(b, d_grid, S, np.linspace(8e-9, 35e-9, 60),
                                   np.linspace(1.01, 1.9, 60), **nz)
        D, G = np.meshgrid(dpg_ax, sg_ax)
        m_d, m_s = post.sum(0), post.sum(1)
        mu_d = (m_d*dpg_ax).sum(); sd_d = np.sqrt((m_d*(dpg_ax-mu_d)**2).sum())
        mu_s = (m_s*sg_ax).sum();  sd_s = np.sqrt((m_s*(sg_ax-mu_s)**2).sum())
        d32 = L.sauter(D, G); mu_32 = (post*d32).sum()
        sd_32 = np.sqrt((post*(d32-mu_32)**2).sum())
        print(f"N={N:3d} | d_pg {mu_d*1e9:.2f}±{sd_d*1e9:.2f} nm | sg {mu_s:.3f}±{sd_s:.3f}"
              f" | d32 {mu_32*1e9:.2f}±{sd_32*1e9:.2f} nm (vrai {L.sauter(dpg0, sg0)*1e9:.2f})"
              f" | corr {np.sum(post*(D-mu_d)*(G-mu_s))/(sd_d*sd_s):+.2f}")
        if N == 1:
            ax[0].plot(t*1e9, b, ".", ms=1, color="0.6", label="données (N=1)")
            ax[0].plot(t*1e9, s_true, "k", lw=1, label="vrai signal")
            ax[0].set(xlabel="t (ns)", ylabel="signal (comptes)", yscale="log",
                      ylim=(1, 2*A)); ax[0].legend()
        a = ax[1+k]
        a.contourf(D*1e9, G, post, levels=30, cmap="viridis")
        a.contour(D*1e9, G, post, levels=sorted([hpd_level(post, .95), hpd_level(post, .68)]),
                  colors="w", linewidths=1)
        a.plot(L.sauter(dpg0, sg0)/np.exp(2.5*np.log(sg_line)**2)*1e9, sg_line, "r--",
               lw=1, label="d32 = d32 vrai")
        a.plot(dpg0*1e9, sg0, "r*", ms=12, label="valeur vraie")
        a.set(xlabel="d_pg (nm)", ylabel="σ_g", xlim=(D.min()*1e9, D.max()*1e9),
              ylim=(G.min(), G.max()), title=f"a posteriori, N={N} tir(s) (68/95 %)")
        a.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig("posterior_grid.png", dpi=120)
