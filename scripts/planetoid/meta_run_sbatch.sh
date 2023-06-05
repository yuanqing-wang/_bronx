for hidden_features in 64 128; do
for embedding_features in 16 32; do
for patience in 5 10 15; do
for factor in 0.5 0.75 0.8 0.9; do
for depth in 3; do
for learning_rate in 1e-2 1e-3 5e-3; do
for weight_decay in 1e-4 1e-5; do
for gamma in 0.7; do
for dropout in 0.5; do
for edge_dropout in 0.2; do

sbatch \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=1 \
	--time=00:10:00 \
	--mem=5GB \
	--job-name=aperol \
    --gres=gpu:v100:1 \
	--output=%A.out \
    --wrap "python run.py \
    --hidden_features $hidden_features \
    --embedding_features $embedding_features \
    --patience $patience \
    --factor $factor \
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
done
done
done
done