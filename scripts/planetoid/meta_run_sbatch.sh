for hidden_features in 64; do
for depth in 2 3 4 5; do
for learning_rate in 1e-2; do
for weight_decay in 1e-4; do
for gamma in 0.7; do
for dropout in 0.5; do

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