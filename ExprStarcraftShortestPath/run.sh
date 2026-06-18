#!/bin/bash
for units in 512 256 128
do
    for layers in 0 1 2 4 8
    do
        
       CUDA_VISIBLE_DEVICES=0 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_0.01.json > logs/${units}_${layers}_0.01.txt & 
       CUDA_VISIBLE_DEVICES=0 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_0.05.json > logs/${units}_${layers}_0.05.txt &
       CUDA_VISIBLE_DEVICES=0 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_0.001.json > logs/${units}_${layers}_0.001.txt &
       CUDA_VISIBLE_DEVICES=0 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_0.005.json > logs/${units}_${layers}_0.005.txt &
       CUDA_VISIBLE_DEVICES=0 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_0.0001.json > logs/${units}_${layers}_0.0001.txt &

       CUDA_VISIBLE_DEVICES=1 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_0.0005.json > logs/${units}_${layers}_0.0005.txt &
       CUDA_VISIBLE_DEVICES=1 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_1e-05.json > logs/${units}_${layers}_1e-05.txt &
       CUDA_VISIBLE_DEVICES=1 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_5e-05.json > logs/${units}_${layers}_5e-05.txt &
       CUDA_VISIBLE_DEVICES=1 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_1e-06.json > logs/${units}_${layers}_1e-06.txt &
       CUDA_VISIBLE_DEVICES=1 python -u  main.py settings/warcraft_shortest_path/${units}_${layers}_5e-06.json > logs/${units}_${layers}_5e-06.txt 
    done
done
