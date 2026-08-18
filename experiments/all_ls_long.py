"""Long-training check for the all-`ls` flow.

The default run (3000 @ 1e-2 + 1000 @ 1e-3) gives ATE +0.057 vs the classical
proportional-odds MLE's +0.055 on the identical data. Since the all-ls flow IS
that model, longer SGD with a finer final learning rate should close the gap —
this script trains 4x longer with an extra 1e-4 phase to verify convergence.
Compare with: uv run python validate_ls.py all_ls_long
"""

from common import run_experiment

if __name__ == "__main__":
    run_experiment(
        "all_ls_long", style="ls", phases=((6000, 1e-2), (6000, 1e-3), (4000, 1e-4))
    )
