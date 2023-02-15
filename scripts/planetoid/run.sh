#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 0:59
#BSUB -n 1

python run.py \
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --num_heads $num_heads \
    --depth $depth \
    --n_samples $n_samples \
    --weight_decay $weight_decay \
    --scale $scale

python upload.py

