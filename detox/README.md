# Detoxification Experiment Archive

This folder contains the detoxification pseudo semantic loss experiments. The
latest implementation appears to be:

- `finetuning-hf-no_trainer_trimmed_improve-fp16-512.py`
- `run_pseudosl_final_512.sh`

The scripts use relative paths, so run them from this directory.

## Important Files

- `finetuning-hf-no_trainer_trimmed_improve-fp16-512.py`: latest 512-token/batch
  pseudo semantic loss fine-tuning script.
- `bad_words.txt`: detox constraint vocabulary source.
- `all_words_trimmed-10_spaces_min.sdd` and `.vtree`: tracked SDD constraint
  artifacts used by the training scripts.
- `exactly_one_trimmed-10_spaces_min.sdd` and `.vtree`: tracked exactly-one
  constraint artifacts used by the training scripts.
- `pypsdd/`: vendored local copy used by the constraint code.
- `data/*.py` and `data/*.ipynb`: toxicity evaluation and analysis utilities.

## Generated Files

Generated outputs are intentionally ignored and were removed from the working
tree during cleanup. This includes logs, model checkpoints, decoded generations,
toxicity-score dumps, WebText/RealToxicity data extracts, pickle outputs, SDD
variants, and notebook checkpoints.

Regenerating experiments will recreate ignored files under paths such as
`logs/`, `hf_models/`, and `data/`.
