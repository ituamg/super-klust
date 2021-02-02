import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from super_klust.super_klust import *
from plot_funcs import plot_boundaries

def main():
    random_state = 111

    X, y = make_moons(n_samples=1000, random_state=random_state, noise=0.15)

    figsize = (4, 2.75)

    spk = SuperKlust(k=2, kmns_n_init=3)
    spk.fit(X, y)
    plot_boundaries(spk, "moons_1", figsize)

    spk = SuperKlust(k=4, kmns_n_init=3)
    spk.fit(X, y)
    plot_boundaries(spk, "moons_2", figsize)

    spk = SuperKlust(k=6, kmns_n_init=3)
    spk.fit(X, y)
    plot_boundaries(spk, "moons_3", figsize)


if __name__ == "__main__":
    main()