<div align="center">

<!-- Logo/Banner placeholder - uncomment and add your image -->
<!-- <img src="assets/banner.png" alt="HyperAgents Banner" width="800"> -->

<h1>HyperAgents</h1>

<p>Self-referential self-improving agents that can optimize for any computable task</p>

<p>
<a href="LICENSE.md"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge" alt="License: CC BY-NC-SA 4.0"></a>
<a href="https://arxiv.org/abs/2603.19461"><img src="https://img.shields.io/badge/arXiv-2603.19461-b31b1b.svg?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
<a href="https://ai.meta.com/research/publications/hyperagents/"><img src="https://img.shields.io/badge/-Blog-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
<a href="https://x.com/jennyzhangzt/status/2036099935083618487"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
</p>

---

</div>

## Setup
```bash
# API keys, put these into .env file
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

```bash
# Install things
sudo dnf install -y python3.12-devel
sudo dnf install -y graphviz graphviz-devel cmake ninja-build bzip2-devel zlib-devel ncurses-devel libffi-devel
```

```bash
# Create virtual environment
python3.12 -m venv venv_nat
source venv_nat/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
# To build the docker container
docker build --network=host -t hyperagents .
```

```bash
# Setup initial agents
bash ./setup_initial.sh
```

## Running HyperAgents

```bash
# See the script for args, and baseline selections
python generate_loop.py --domains <domain>
```

By default, outputs will be saved in `outputs/` directory.

## IMO grading experiment records

Each attempt keeps its existing generation ID (`gen_0`, `gen_1`, ...,
`gen_30`); round 0 is the initial baseline. New IMO runs write:

```text
gen_1/
  metadata.json          # parent, state, models, usage, stage summaries
  model_patch.diff       # changes relative to parent; empty if no changes
  meta_agent_chat.md
  generate.log
  pre_eval/              # fixed 10 training examples
  train_eval/            # 100 training examples, reusing screening predictions
  val_eval/              # 100 validation examples
```

Each evaluation directory contains `stage_status.json`; attempted stages also
keep `predictions.csv`, `evaluation.jsonl`, and per-example model traces in
`agent_evals/`. Successful stages add `report.json`. Predictions are checkpointed after each example;
failed stages retain partial results. Unreached stages have a status file with
the reason. Raw container exports and request usage logs are also retained.
Failed, screening-rejected, and empty-diff attempts remain in the archive.

Regenerate figures from an existing run (no model requests):

```bash
python -m analysis.visualize_archive --domains imo_grading --path outputs/generate_RUN --svg
python -m analysis.plot_progress --domains imo_grading --path outputs/generate_RUN --svg
```

The evolution tree includes isolated nodes, distinguishes failure and screening
rejection, and exports a JSON file containing node states and parent edges.
Graphviz is preferred; a deterministic layered layout is used if unavailable.
Validation plots use only successful complete 100-example results, preserve
every attempt on the x-axis, and mark missing measurements without assigning
them zero accuracy. Per-candidate scores and best-so-far scores are exported
as JSON and CSV. Readers also accept legacy `imo_grading_eval*` directories.

## File Structure
- `agent/` code for using foundation models
- `analysis/` scripts used for plotting and analysis
- `domains/` code for each domain
- `utils/` common code used in the repo
- `run_meta_agent.py` script to help run the meta agent and get the diffs
- `meta_agent.py` main implementation of the meta agent
- `task_agent.py` main implementation of the task agent
- `generate_loop.py` entry point for running the algorithm

## Logs from Experiments

The experiment logs can be downloaded here: https://drive.google.com/drive/folders/164fKQWgLM18foOzSnpv0F_I3TNpX8u8-?usp=sharing

## Safety Consideration
> [!WARNING]  
> This repository involves executing untrusted, model-generated code. We strongly advise users to be aware of the associated safety risks. While it is highly unlikely that such code will perform overtly malicious actions under our current settings and with the models we use, it may still behave destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.

## Citing
If you find this project useful, please consider citing:
```bibtex
@misc{zhang2026hyperagents,
      title={Hyperagents}, 
      author={Jenny Zhang and Bingchen Zhao and Wannan Yang and Jakob Foerster and Jeff Clune and Minqi Jiang and Sam Devlin and Tatiana Shavrina},
      year={2026},
      eprint={2603.19461},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.19461}, 
}
```

