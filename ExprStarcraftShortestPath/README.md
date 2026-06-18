# Running ExprStarcraftShortestPath

This repo is set up around the `warcraft_shortest_path` experiment in `main.py`.

## Setup

Install dependencies with either:

- `pipenv install`
- `pip install -r requirements.txt`

The code expects the Warcraft data under:

- `data/warcraft_shortest_path/12x12`

The current pseudo-SL code also expects these constraint files to exist:

- `data/warcraft_shortest_path/12x12/constraint_trimmed.sdd`
- `data/warcraft_shortest_path/12x12/constraint_trimmed.vtree`

## Run one experiment

Baseline:

```bash
python main.py settings/pseudo_sl/base.json
```

Pseudo semantic loss:

```bash
python main.py settings/pseudo_sl/sl.json
```

You can also run any specific weight directly, for example:

```bash
python main.py settings/pseudo_sl/sl_0.001.json
python main.py settings/pseudo_sl/sl_0.005.json
python main.py settings/pseudo_sl/sl_0.01.json
```

## Run the pseudo-SL sweep

Create the log directory first:

```bash
mkdir -p logs
```

Then run:

```bash
bash run_sl_experiments.sh
```

## Outputs

- model outputs and metrics are written under the `model_dir` specified in the chosen settings file
- sweep logs are written to `logs/`
