for hidden_features in 32 64 128; do
for depth in 2 3 4 5; do
for learning_rate in 1e-2; do
for weight_decay in 1e-3; do
for dropout in 0.5; do
for embedding_features in 32 64 128; do
for num_heads in 2 4 8 16; do
for patience in 5; do
for factor in 0.5; do

sbatch \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=1 \
	--time=00:10:00 \
	--job-name=aperol \
    --gres=gpu:1 \
	--output=%A.out \
    --wrap "python run.py \
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --depth $depth \
    --weight_decay $weight_decay \
    --dropout $dropout \
    --embedding_features $embedding_features \
    --factor $factor \
    --patience $patience\
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
