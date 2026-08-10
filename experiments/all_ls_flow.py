"""All-`ls` causal flow (this package's analog of the original `md_dag_ls` run).

Every edge is a linear shift, so each node-conditional is a classical
(proportional-odds / linear transformation) model.

Default data is the public synthetic cohort (`magic-mrclean/nl`); pass a source
to switch, e.g. the private clinical data:

    uv run python all_ls_flow.py                  # synthetic (default)
    uv run python all_ls_flow.py magic-mrclean/ls # synthetic, linear variant
    uv run python all_ls_flow.py magic            # private clinical cohort
"""

from common import run_experiment, run_name, source_arg

if __name__ == "__main__":
    source = source_arg(__doc__)
    run_experiment(run_name("all_ls", source), style="ls", source=source)
