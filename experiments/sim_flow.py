"""The simulation storyline: known-truth recovery on the synthetic cohort.

On a chosen variant of `magic-mrclean` it fits BOTH the all-`ls` flow and the
flexible (`ci`/`cs`) flow, and reports each estimated ATE against the known true
ATE from `truth.json` — plus the naive (confounded) observational contrast.

Expected reading:
  - `nl` variant: the all-`ls` model is misspecified (it must collapse the
    age-varying treatment effect to one constant) and is biased; the flexible
    flow recovers the true ATE. Both beat the naive observational difference,
    which is badly confounded.
  - `ls` variant: both flows and the true ATE coincide (the DGP is all-`ls`).

Usage: uv run python sim_flow.py [ls|nl]   (default nl)
"""

import argparse
import logging

from common import run_experiment

if __name__ == "__main__":
    # fit() reports progress on the tramdag.flow logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "variant",
        nargs="?",
        default="nl",
        choices=("ls", "nl"),
        help="magic-mrclean variant (default: nl)",
    )
    variant = parser.parse_args().variant
    source = f"magic-mrclean/{variant}"
    print(f"\n################ all-ls flow on {source} ################")
    run_experiment(f"sim_{variant}_ls", style="ls", source=source)
    print(f"\n################ flexible flow on {source} ################")
    run_experiment(f"sim_{variant}_flex", style="flexible", source=source)
