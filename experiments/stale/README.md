# Parked experiments

Parked during the 0.4 cleanup and **not maintained against the current
API**. The maintained entry points are `sim_flow.py` (the headline
storyline) and `validate_ls.py` (the classical-equivalence check), both
one directory up. The paper replications (`paper_*.py`) moved back up: they are
maintained and run against the current API. Restore anything else from
here by migrating it to the transformation syntax first.

Note: `nihss6_flow.py` needs more than the syntax port — it imports
`run_name` and `source_arg`, which no longer exist in `common.py`.
