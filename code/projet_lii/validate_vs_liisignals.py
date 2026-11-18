"""Compare T(t) du modèle Python à LIISignals (C++), sublimation coupée.
Préparation de la référence (une fois) :
  cp -r LIISignals-master/src ref/ ; cp LIISignals-master/examples/01/main.cpp ref/
  sed -i 's/getchar();//' ref/main.cpp
  g++ -O2 -std=c++11 -Iref/src ref/main.cpp -o ref/lii
  copier examples/01/input dans ref/, mettre beta (mass accommodation) à 0
  cd ref && ./lii      -> SolutionMonodispersed.out
"""
import numpy as np
import lii_model as m

r = np.loadtxt("ref/SolutionMonodispersed.out", skiprows=1)
t, Tref = r[:, 0], r[:, 2]
T = m.temperature(21e-9, t)
e = np.abs(T - Tref)
print("Tmax py %.1f | ref %.1f" % (T.max(), Tref.max()))
print("max |dT| = %.2f K (rel %.1e)" % (e.max(), (e / Tref).max()))
