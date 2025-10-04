# fem-ipc

A simple Incremental Potential Contact implementation.

- Stable Neo-Hookean Energy for soft bodies dynamics
- Moving boundary collisions

The constraint sets are computed by brute force, which is a dominant bottleneck.

## References

- Minchen Li, Zachary Ferguson, Teseo Schneider, Timothy Langlois, Denis Zorin, Daniele Panozzo, Chenfanfu Jiang, and Danny M. Kaufman. 2020. Incremental potential contact: intersection-and inversion-free, large-deformation dynamics. ACM Trans. Graph. 39, 4, Article 49 (August 2020), 20 pages. https://doi.org/10.1145/3386569.3392425
- Breannan Smith, Fernando De Goes, and Theodore Kim. 2018. Stable Neo-Hookean Flesh Simulation. ACM Trans. Graph. 37, 2, Article 12 (April 2018), 15 pages. https://doi.org/10.1145/3180491
