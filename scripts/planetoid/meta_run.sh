for hidden_features in 8 16 32 64; do
for learning_rate in 1e-2 1e-3 1e-4 1e-5; do
for weight_decay in 1e-2 1e-4 1e-6 1e-8 1e-10; do
for num_heads in 1 2 4 8; do
for dropout in 0.3 0.5 0.7; do


bsub \
    -q gpuqueue \
    -o %J.stdout \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=5] span[ptile=1]" \
    -W 0:30 \
    -R V100 \
    -n 1 \
    python run.py \
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --num_heads $num_heads \
    --dropout $dropout

done
done
done
done
done
