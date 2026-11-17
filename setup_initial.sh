#!/bin/bash
set -euo pipefail
# Data preparation stays on the trusted host. Round 0 evaluates train and val
# inside Docker; no test evaluation is performed during initialization.
python -m domains.imo.curate_subsets
python -m generate_loop --domains imo_grading --max_generation 30 "$@"
