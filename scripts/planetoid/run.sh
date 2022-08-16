#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=5] span[ptile=1]"
#BSUB -W 0:20
#BSUB -n 1

python run.py \
    --hidden_features $hidden_features \
    --learning_rate $learning_rate \
    --depth $depth \
    --residual $residual \
    --weight_decay $weight_decay \
    --semantic_weight $semantic_weight \
    --num_heads $num_heads \
    --a_h_dropout $a_h_dropout \
    --a_x_dropout $a_x_dropout \
    --fc_dropout $fc_dropout

python upload.py

