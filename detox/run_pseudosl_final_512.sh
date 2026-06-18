for W in 0 #1 0.1 0.5 0.01 0.05 #1 #0.005 0.001
do
	CUDA_VISIBLE_DEVICES=0,1,2 python -u finetuning-hf-no_trainer_trimmed_improve-fp16-512.py --model_name_or_path gpt2 --train_file non_toxic.json --per_device_train_batch_size 128 --per_device_eval_batch_size 8 --output_dir hf_models/final/ --num_train_epochs 3 --block_size 100 --gradient_accumulation_steps 4 --checkpointing_steps 10 --validation_file toxic_data.json --sl_weight $W --learning_rate 2e-5 --seed 0 &> logs/final/$W\_0.00002\_512.log
done
