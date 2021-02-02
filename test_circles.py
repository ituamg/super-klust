import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from super_klust.super_klust import *
from plot_funcs import plot_boundaries

def main():
    random_state = 111

    X, y = make_circles(n_samples=1000, random_state=random_state, noise=0.05)
    
    figsize = (4, 4.5)

    spk = SuperKlust(k=6, kmns_n_init=3)
    spk.fit(X, y)
    plot_boundaries(spk, "circles_1", figsize)

    spk = SuperKlust(k=8, kmns_n_init=3)
    spk.fit(X, y)
    plot_boundaries(spk, "circles_2", figsize)

    spk = SuperKlust(k=12, kmns_n_init=3)
    spk.fit(X, y)
    plot_boundaries(spk, "circles_3", figsize)


if __name__ == "__main__":
    main()