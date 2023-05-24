for hidden_features in 64 128 256; do
for depth in 2 3 4 5; do
for learning_rate in 1e-2 5e-3; do
for weight_decay in 1e-4 5e-4; do
for gamma in 0.0 0.2 0.4; do
for dropout in 0.0 0.5; do

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
    --depth $depth \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --dropout $dropout \
"

done
done
done
done
done
done