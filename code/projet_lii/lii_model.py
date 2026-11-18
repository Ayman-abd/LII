"""Modèle LII basse fluence, conduction en régime moléculaire libre (monodisperse).
Bilan : m cp dT/dt = Q_abs - Q_cond - Q_rad  (pas de sublimation).
Propriétés reprises de LIISignals (Hofmann 2007) pour pouvoir comparer.
"""
import numpy as np
from scipy.integrate import solve_ivp

h, c, kB, R = 6.62607004e-34, 299792458.0, 1.38064852e-23, 8314.4598
sigmaSB = 5.670367e-8

P = dict(
    Em=0.4, lam_ex=1064e-9, F0=0.1e4,          # E(m), laser (m), fluence (J/m2)
    fwhm=7e-9, t_laser=60e-9,                  # impulsion gaussienne
    alpha=0.3, Tg=1700.0, p=1e5, Mg=28.0,      # gaz (N2)
    rho=1860.0, lam_det=550e-9, theta=1.0,     # suie, détection
)

def cp_soot(T):                                 # J/kg/K
    return 1878.0 + 0.1082 * T - 1.5149e8 / T**2

def cp_N2(T):                                   # J/kg/K (fit CRECK, comme LIISignals)
    T = np.minimum(T, 3800.0)
    lo = 1.143621e3 + T*(-7.243422e-1 + T*(1.588338e-3 + T*(-1.114793e-6 + T*2.738496e-10)))
    hi = 8.051735e2 + T*(5.649814e-1 + T*(-2.535527e-4 + T*(5.466132e-8 - T*4.591919e-12)))
    return np.where(T <= 1050.0, lo, hi)

def gamma(T, Mg):
    Cp = cp_N2(T) * Mg
    return Cp / (Cp - R)

def q_laser(t, P):                              # W/m2
    s = P["fwhm"] / (2*np.sqrt(2*np.log(2)))
    return P["F0"] / (s*np.sqrt(2*np.pi)) * np.exp(-(t - P["t_laser"])**2 / (2*s**2))

def rhs(t, T, d, P):
    Tg, Mg = P["Tg"], P["Mg"]
    Q_abs = np.pi**2 * d**3 / P["lam_ex"] * P["Em"] * q_laser(t, P)
    # gamma* moyen (Simpson entre Tg et Tp)
    I = (1/(gamma(Tg, Mg)-1) + 4/(gamma((Tg+T)/2, Mg)-1) + 1/(gamma(T, Mg)-1)) / 6
    gs = 1 + 1/I
    Q_cond = P["alpha"]*np.pi*d**2*P["p"]/8*np.sqrt(8*R*Tg/np.pi/Mg)*(gs+1)/(gs-1)*(T/Tg - 1)
    Q_rad = np.pi*d**2*P["theta"]*sigmaSB*(T**4 - Tg**4)
    m = P["rho"]*np.pi/6*d**3
    return (Q_abs - Q_cond - Q_rad) / (m*cp_soot(T))

def temperature(d, t, P=P):
    """T(t) d'une particule de diamètre d (m) sur la grille t (s)."""
    sol = solve_ivp(rhs, (t[0], t[-1]), [P["Tg"]], t_eval=t, args=(d, P),
                    method="LSODA", rtol=1e-8, atol=1e-6, max_step=0.5e-9)
    return sol.y[0]

def signal(d, T, P=P):
    """Incandescence (u.a.) à lam_det : emissivité Rayleigh ∝ d/λ, surface ∝ d²."""
    lam = P["lam_det"]
    return d**3 / lam**6 / np.expm1(h*c/(lam*kB*T))

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.linspace(0, 2e-6, 4001)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for d in (10e-9, 21e-9, 40e-9):
        T = temperature(d, t)
        S = signal(d, T)
        ax[0].plot(t*1e9, T, label=f"{d*1e9:.0f} nm")
        ax[1].plot(t*1e9, S/S.max())
    ax[0].set(xlabel="t (ns)", ylabel="T_p (K)"); ax[0].legend()
    ax[1].set(xlabel="t (ns)", ylabel="signal normalisé", yscale="log", ylim=(1e-3, 1.2))
    fig.tight_layout(); fig.savefig("mono.png", dpi=120)
