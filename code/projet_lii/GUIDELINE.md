# Projet 5A ONERA — Inversion bayésienne TiRe-LII (taille des suies)

Encadrant : Romain Ceolato (ONERA Toulouse). Volet PINN demandé en exploration.

## 1. Le problème en bref
- Un laser (ns) chauffe des billes de suie (monomères 10–50 nm) jusqu'à ~3500 K ; elles refroidissent
  par conduction vers le gaz en ~1 µs. On mesure leur incandescence S(t).
- Petite bille = refroidissement rapide (chaleur stockée ∝ d³, perdue ∝ d²  =>  vitesse ∝ 1/d).
- Les tailles suivent une log-normale (d_pg = médiane, σ_g = largeur) : **les 2 seules inconnues**.
- Modèle direct (« modèle LII basse fluence, conduction en régime moléculaire libre », Michelsen 2007) :
  `m cp dT/dt = Q_abs(t) − Q_cond(T) − Q_rad(T)`, sans sublimation ;
  signal = ∫ p(d; d_pg, σ_g) · d³ · Planck(λ_det, T(t,d)) dd.
- Problème mal posé : des distributions de même diamètre de Sauter
  `d32 = d_pg·exp(2.5 ln²σ_g)` donnent des signaux quasi identiques -> vallée dans le plan (d_pg, σ_g).
- Couplage α–d : en conduction pure, seul α/d compte (Sipkens 2015, Daun 2007) -> une erreur
  sur α biaise d proportionnellement.

## 2. Ce qui est fait
| Fichier | Contenu | Validation |
|---|---|---|
| `lii_model.py` | EDO monodisperse (absorption gaussienne, conduction FM, rayonnement), signal | écart 0,1 K vs LIISignals (β=0), `mono.png` |
| `validate_vs_liisignals.py` | comparaison au code C++ | — |
| `lognormal.py` | bibliothèque T(t,d) sur 120 tailles résolue une fois, signal polydisperse = somme pondérée (~1 ms) | convergence 120 vs 240 tailles : 1e-10 ; `poly.png` montre la dégénérescence d32 |
| `mcmc.py` | emcee sur (ln d_pg, ln σ_g, ln A), prior uniforme en ln, vraisemblance vectorisée ; 4 ensembles × 32 marcheurs × 6000 pas, départ = Nelder-Mead depuis un point aléatoire de la boîte | A connue : identique à la grille 2D ; A inconnue : identique à une grille 3D marginalisée sur A (écarts < 0,5 % sur moyennes et σ) ; R̂ < 1,01, ESS > 14 000 ; `mcmc_posterior.png` |
| `twin.py` | expériences jumelles : inversion par grille 3D (plan (ln d_pg, ln σ_g) blanchi le long de la vallée + A intégrée en ln A, 2 passes), couverture 68/95 % ; reprise possible (`twin_parts/`), `--plot` refait la figure | `--check` : identique au MCMC à 1e-3 près ; `twin_coverage.png` |
| `grid_posterior.py` | bruit Sipkens 2017 `Var = (γ² + θ s + τ² s²)/N`, vraisemblance gaussienne, a posteriori sur grille 2D avec zoom auto | retrouve (20 nm, 1,3) ; corrélation −0,99 le long de d32 constant |

Résultats grille (d_pg=20 nm, σ_g=1,3) :
- N=100 tirs : d_pg 19,98±0,09 nm ; σ_g 1,300±0,003 ; d32 23,74±0,04 nm
- N=1 tir    : d_pg 19,1±0,9 nm ;  σ_g 1,32±0,03 ;   d32 23,3±0,4 nm

Résultats MCMC, mêmes données (prior uniforme en ln) :
| cas | d_pg (nm) | σ_g | d32 (nm) | A (vrai 1000) | τ_int | R̂ max |
|---|---|---|---|---|---|---|
| N=100, A connue   | 19,98±0,09 | 1,300±0,003 | 23,74±0,04 | — | ~33 | 1,006 |
| N=100, A inconnue | 20,00±0,15 | 1,300±0,004 | 23,75±0,08 | 999,7±2,2 | ~38 | 1,008 |
| N=1, A connue     | 19,1±0,9   | 1,325±0,029 | 23,29±0,39 | — | ~32 | 1,007 |
| N=1, A inconnue   | 19,0±1,4   | 1,327±0,040 | 23,2±0,7   | 1002±22 | ~50 | 1,009 |
-> Rendre A inconnue élargit l'incertitude d'environ ×1,5 sur d_pg et ×2 sur d32.
Remarques : (i) un départ uniforme dans toute la boîte laisse des marcheurs coincés au bord
(σ_g≈1,01, acceptation <1 %) d'où l'initialisation par optimisation ; (ii) les 4 optimisations
depuis des points aléatoires tombent sur le même mode (pas de multimodalité vue) ; (iii) la grille de
`grid_posterior.py` a un prior uniforme en variables physiques, celle de `mcmc.py` en ln (jacobien
1/(d_pg σ_g A)) : différence invisible ici ; (iv) `grid_posterior.py` prend ~5 min (pas ~1 min) ;
(v) borne haute du prior sur d_pg passée de 40 à 60 nm (la postérieure du cas 30 nm/1,15 était
coupée à 40 nm) ; sans effet sur le cas 20 nm.

Expériences jumelles (`twin.py`, A inconnue, 300 réalisations par cas, couples vrais
20 nm/1,3 ; 12 nm/1,6 ; 30 nm/1,15 ; N = 1 et 100). Tolérance ±2σ binomiale : 68 % ± 5,4, 95 % ± 2,5.
- **τ indépendant à la génération (modèle d'inversion juste)** : couverture correcte. 51 des 60
  valeurs dans la tolérance (≈ 57 attendues par hasard) ; les écarts restent ≤ 4,3 pts
  (pire : d_pg 95 % = 90,7 % pour N=1, 20 nm). Le cas 30 nm/1,15 en N=1 sur-couvre un peu
  (≈ 98 % pour 95 %) : la postérieure est tronquée par la borne σ_g ≥ 1,01 du prior.
  z(d32) = (moyenne − vrai)/σ : moyenne ≈ 0, écart-type 0,96–1,11 -> incertitudes bien calibrées.
- **τ commun à toute la courbe à la génération (modèle d'inversion faux)** :
  - A : couverture effondrée (7–12 % au lieu de 68 %, 14–19 % au lieu de 95 %). Le facteur commun
    (±20 %/√N) est absorbé par A, alors que la vraisemblance croit A connue à ~2 %/√N près.
  - d_pg, σ_g, d32 : **sur-couverture** (d32 : 88–94 % au lieu de 68 %), z(d32) écart-type ≈ 0,56–0,61
    -> les incertitudes sont environ 1,7× trop larges, sans biais : le terme τ²s² par point gonfle
    la variance supposée alors que les données n'ont pas ce bruit point à point.
  - À comprendre : en N=1 la région HPD 2D reste proche du nominal (60–67 % / 93–96 %) alors que
    les marginales 1D sur-couvrent. Hypothèse : sur-largeur surtout le long de la vallée, et
    région 2D mal orientée. À vérifier.
  - Conclusion pratique : si les fluctuations tir à tir sont vraiment communes à la courbe (énergie
    laser, alignement), il faut retirer τ²s² de la variance point à point et laisser A l'absorber
    (ou covariance de rang 1 τ² s sᵀ/N). A n'est alors plus interprétable comme quantité physique.
- Coût : 3600 inversions en 2 h 34 sur 10 processus (contention des threads BLAS ; corrigé
  depuis : 1 thread par processus, progression affichée, paquets sauvegardés).

## 3. Hypothèses provisoires à lever
1. **Bruit** : γ=10, θ=1, τ=0,2, pic A=1000 comptes -> ordres de grandeur, à recaler sur Sipkens 2017
   (Appl. Opt. 56, 8436 ; code : github.com/tsipkens/pgm-error). PDF pas encore récupéré.
2. ~~**Amplitude A supposée connue**~~ -> **fait** dans `mcmc.py` (3e paramètre, cf. tableau).
3. **τ traité indépendant à chaque instant** alors qu'il multiplie toute la courbe d'un tir.
   Testé dans `twin.py` : si le vrai bruit est commun, d_pg/σ_g restent non biaisés mais leurs
   incertitudes sont ~1,7× trop larges, et celle de A est fausse. Reste à trancher sur données
   réelles / Sipkens 2017, puis à corriger le modèle de bruit (cf. §2).
4. **Rayonnement corps noir (θ=1)** repris de LIISignals -> remplacer par l'émissivité Rayleigh
   (effet faible).
5. Paramètres = exemple LIISignals (1064 nm, 0,1 J/cm², Tg 1700 K, α 0,3, E(m) 0,4). Vérifier
   que la sublimation est bien négligeable, sinon baisser la fluence (ex. 0,05 J/cm²).
6. Ne pas utiliser le mode polydisperse de LIISignals comme référence (bug : sommes partielles).

## 4. Étapes suivantes
1. ~~**MCMC (emcee)**~~ **fait** (`mcmc.py`). Pour les expériences jumelles, réutiliser
   `Model.shape` + `make_log_prob` (≈ 2–3 min par inversion à 6000 pas ; 1500–2000 pas suffisent
   vu τ ≈ 40, soit ~40 s ; ou la grille 3D vectorisée `grid_reference`, quelques secondes).
2. ~~**Expériences jumelles**~~ **fait** (`twin.py`, résultats au §2). Suites possibles :
   inverser le cas « τ commun » avec un modèle de bruit corrigé (sans τ²s² ou rang 1) et
   vérifier que la couverture redevient nominale ; comprendre l'écart HPD 2D / marginales.
3. **Sensibilité α** : générer avec α_vrai, inverser avec α_supposé ∈ [0,2 ; 0,4] ; tracer le biais
   relatif sur d_pg et d32 vs α_supposé/α_vrai (attendu ≈ proportionnel).
4. **PINN (exploration)** :
   - PINN paramétrique T(τ; d, α) en variables adimensionnées, départ au pic, temps log,
     condition initiale imposée en dur -> substitut différentiable -> NUTS (numpyro) ;
     comparer la postérieure substitut vs exacte.
   - Option : PINN inverse (d entraînable) comme contre-exemple (pas d'incertitude).
   - Honnêteté : pour une EDO 1D, solve_ivp reste plus précis/rapide ; l'intérêt est la
     différentiabilité et l'amortissement.
5. **Rapport** : biblio, modèle + validation, bruit, grille vs MCMC, couverture, α, PINN.

## 5. Ordre de lecture
1. Michelsen 2015 (`liiArticle.pdf`) : intro + principes/bilan d'énergie.
2. Michelsen 2007 (`s00340-007-2619-5`) : équations du modèle, cas 0,05 J/cm² ; 10 modèles comparés.
3. Daun 2007 (`s00340-007-2585-y`) : mal-posé, vallée d32.
4. Sipkens 2022 (`s00340-022-07769-z`) : §2.3–2.4 (cadre bayésien).
5. Sipkens 2015 (`APB2015…RSG2.pdf`) : couplage α/d, annexe A (MCMC, marginalisation).
6. Sipkens 2019 (`s00340-019-7251-7`) : polydispersité, modèle simplifié (plus tard).
7. À récupérer : Sipkens 2017 (bruit), Hadwin 2018 (Kalman, réf. [4] du sujet).

## 6. Lancer
```
pip install numpy scipy matplotlib emcee
python lii_model.py        # mono.png
python lognormal.py        # poly.png
python grid_posterior.py   # posterior_grid.png (~5 min)
python mcmc.py             # mcmc_posterior.png (~10 min)
python twin.py --check     # inverseur grille vs MCMC (~20 s)
python twin.py             # twin_coverage.png (3600 inversions ; reprise auto via twin_parts/)
python twin.py --plot      # tableau + figure depuis twin_results.npy
```
