# Parked notebooks

Parked during the 0.4 cleanup and **not maintained against the current
API** — none of them runs as written. All three use the removed `terms=`
keyword; `transforms_tram_dag.py` also uses node-level
`transform=`/`transform_kwargs=`, and `classical_fit_tram_dag.py` imports
`tramdag.simulations` (which left the package in 0.4) and reads `data/`
at its old top-level path (now `experiments/data/`). They are excluded
from lint for that reason. Restore one by moving it back up, migrating it
to the transformation syntax and repointing its data; the maintained
examples are
[`intro_tram_dag.py`](../intro_tram_dag.py) and
[`demo_tram_dag_colab.py`](../demo_tram_dag_colab.py).
