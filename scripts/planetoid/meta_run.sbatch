for hidden_features in 128; do
for learning_rate in 1e-2; do
for weight_decay in 1e-4; do
for num_heads in 4; do
for dropout0 in 0.6; do
for dropout1 in 0.6; do
for gamma in 0.2; do
for factor in 0.8; do
for patience in 10; do
for gain in 0.1; do


sbatch \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=1 \
	--time=00:30:00 \
	--mem=5GB \
	--job-name=aperol \
    --gres=gpu:v100:1 \
	--output=%A.out \
    --wrap "python run.py \
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --num_heads $num_heads \
    --dropout0 $dropout0 \
    --dropout1 $dropout1 \
    --gamma $gamma \
    --factor $factor \
    --patience $patience \
    --gain $gain"

done
done
done
done
done
done
done
done
done
done
