for hidden_features in 64 128; do
for depth in 2 3; do
for learning_rate in 1e-2 1e-3; do
for weight_decay in 1e-5 1e-4; do
for gamma in 1.0; do
for patience in 5 10; do
for factor in 0.5 0.8; do

sbatch \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=1 \
	--time=00:30:00 \
	--mem=5GB \
	--job-name=aperol \
    --gres=gpu:1 \
	--output=%A.out \
    --wrap "python run.py \
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --depth $depth \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --factor $factor \
    --patience $patience\
"

done
done
done
done
done
done
done
