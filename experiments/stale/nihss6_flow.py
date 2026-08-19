"""Flexible causal flow — the configuration featured in the paper (`nihss6`).

Per-edge terms as in nihss6/configuration.json: Age enters every child via a
complex intercept ('ci'), mRS_pre via linear shifts ('ls'), NIHSSa via complex
shifts ('cs'), T -> mRS_3m via a linear shift ('ls').

Default data is the public synthetic cohort (`magic-mrclean/nl`); pass a source
to switch, e.g. `magic` for the private clinical cohort:

    uv run python nihss6_flow.py                  # synthetic (default)
    uv run python nihss6_flow.py magic            # private clinical cohort
"""

from common import run_experiment, run_name, source_arg

if __name__ == "__main__":
    source = source_arg(__doc__)
    run_experiment(run_name("nihss6", source), style="flexible", source=source)
