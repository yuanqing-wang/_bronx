for hidden_features in 256; do
for learning_rate in 1e-2 5e-3; do
for weight_decay in 1e-4 5e-4; do
for num_heads in 4; do
for dropout0 in 0.2 0.4 0.6; do
for dropout1 in 0.2 0.4 0.6; do
for gamma in 0.2 0.4 0.6 0.8; do




sbatch \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=1 \
	--time=00:15:00 \
	--mem=2GB \
	--job-name=aperol \
	--output=%A.out \
	--gres=gpu:v100:1 \
    --wrap "python run.py \
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --num_heads $num_heads \
    --dropout0 $dropout0 \
    --dropout1 $dropout1 \
    --gamma $gamma"

done
done
done
done
done
done
done
