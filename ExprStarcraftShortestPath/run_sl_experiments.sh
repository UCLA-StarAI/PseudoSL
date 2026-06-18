 #!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u main.py settings/pseudo_sl/sl_0.001.json > logs/psl_0.001.json
CUDA_VISIBLE_DEVICES=0 python -u main.py settings/pseudo_sl/sl_0.005.json > logs/psl_0.005.json
CUDA_VISIBLE_DEVICES=0 python -u main.py settings/pseudo_sl/sl_1e-4.json > logs/psl_1e-4.json
CUDA_VISIBLE_DEVICES=0 python -u main.py settings/pseudo_sl/sl_5e-4.json > logs/psl_5e-4.json
CUDA_VISIBLE_DEVICES=0 python -u main.py settings/pseudo_sl/sl_1e-5.json > logs/psl_1e-5.json
CUDA_VISIBLE_DEVICES=0 python -u main.py settings/pseudo_sl/sl_5e-5.json > logs/psl_5e-5.json
