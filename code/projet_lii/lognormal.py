"""Signal LII polydisperse (log-normale) à partir du modèle monodisperse.
Idée : T(t, d) ne dépend pas de (d_pg, sigma_g) -> on calcule une fois une
bibliothèque S(t, d) sur une grille de tailles, puis chaque signal = somme pondérée.
"""
import numpy as np
from scipy.integrate import solve_ivp
import lii_model as m

def library(t, d_grid, P=m.P):
    """Résout l'EDO pour toutes les tailles d'un coup (vecteur d'états)."""
    sol = solve_ivp(lambda tt, T: m.rhs(tt, T, d_grid, P), (t[0], t[-1]),
                    np.full(d_grid.size, P["Tg"]), t_eval=t,
                    method="RK45", rtol=1e-8, atol=1e-6, max_step=0.5e-9)
    T = sol.y                                   # (n_d, n_t)
    S = m.signal(d_grid[:, None], T, P)         # (n_d, n_t)
    return T, S

def lognormal_pdf(d, dpg, sg):
    ls = np.log(sg)
    return np.exp(-(np.log(d/dpg))**2 / (2*ls**2)) / (d*ls*np.sqrt(2*np.pi))

def poly_signal(dpg, sg, d_grid, S):
    """Signal normalisé au pic (l'amplitude absolue est inconnue en pratique)."""
    w = lognormal_pdf(d_grid, dpg, sg)
    s = np.trapezoid(w[:, None] * S, d_grid, axis=0)
    return s / s.max()

def sauter(dpg, sg):
    return dpg * np.exp(2.5*np.log(sg)**2)

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.linspace(0, 2e-6, 2001)
    d_grid = np.geomspace(2e-9, 200e-9, 120)
    T, S = library(t, d_grid)

    # Contrôle 1 : bibliothèque vs solveur monodisperse
    i = np.argmin(abs(d_grid - 21e-9))
    print("écart T lib/mono : %.2f K" % abs(T[i] - m.temperature(d_grid[i], t)).max())
    # Contrôle 2 : convergence en nombre de tailles
    d2 = np.geomspace(2e-9, 200e-9, 240); _, S2 = library(t, d2)
    a, b = poly_signal(20e-9, 1.4, d_grid, S), poly_signal(20e-9, 1.4, d2, S2)
    print("écart signal 120 vs 240 tailles : %.1e" % abs(a - b).max())

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # (a) même d_pg, largeurs différentes
    for sg in (1.05, 1.2, 1.4, 1.6):
        ax[0].plot(t*1e9, poly_signal(20e-9, sg, d_grid, S), label=f"σg={sg:.1f}")
    ax[0].set(title="d_pg = 20 nm", xlabel="t (ns)", ylabel="signal normalisé",
              yscale="log", ylim=(1e-3, 1.2)); ax[0].legend()
    # (b) même diamètre de Sauter d32 = 30 nm -> signaux quasi identiques
    for sg in (1.1, 1.3, 1.5):
        dpg = 30e-9 / np.exp(2.5*np.log(sg)**2)
        ax[1].plot(t*1e9, poly_signal(dpg, sg, d_grid, S),
                   label=f"σg={sg}, d_pg={dpg*1e9:.1f} nm")
    ax[1].set(title="même d32 = 30 nm", xlabel="t (ns)", yscale="log", ylim=(1e-3, 1.2))
    ax[1].legend()
    fig.tight_layout(); fig.savefig("poly.png", dpi=120)
