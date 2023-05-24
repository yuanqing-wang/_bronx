for hidden_features in 64; do
for depth in 3; do
for learning_rate in 1e-2; do
for weight_decay in 1e-5; do
for gamma in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
for dropout in 0.5; do
for edge_drop in 0.2; do
for embedding_features in 16 32 64; do
for num_heads in 1 2 4 8; do

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
    --dropout $dropout \
    --edge_drop $edge_drop \
    --embedding_features $embedding_features \
    --num_heads $num_heads \
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
