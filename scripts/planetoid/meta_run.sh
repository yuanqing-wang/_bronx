for hidden_features in 256; do
for learning_rate in 1e-2; do
for weight_decay in 1e-4; do
for num_heads in 4; do
for dropout0 in 0.6; do
for dropout1 in 0.6; do
for gamma in 0.2 0.4 0.6 0.8 1.0; do

bsub \
    -q gpuqueue \
    -o %J.stdout \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=5] span[ptile=1]" \
    -W 0:15 \
    -n 1 \
    python run.py \
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --num_heads $num_heads \
    --dropout0 $dropout0 \
    --dropout1 $dropout1 \
    --gamma $gamma

done
done
done
done
done
done
done
