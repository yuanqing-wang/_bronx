for hidden_features in 64; do
for depth in 3; do
for learning_rate in 1e-2; do
for weight_decay in 1e-5; do
for gamma in 0.7; do
for dropout in 0.5; do
for edge_drop in 0.2 0.4 0.6 0.8; do
for embedding_features in 16; do
for num_heads in 1 2 4 8; do

bsub \
    -q gpuqueue \
    -o %J.stdout \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=5] span[ptile=1]" \
    -W 0:10 \
    python run.py
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --depth $depth \
    --weight_decay $weight_decay \
    --gamma $gamma \
    --dropout $dropout \
    --edge_drop $edge_drop \
    --embedding_features $embedding_features \
    --num_heads $num_heads

done
done
done
done
done
done
done
done
done

