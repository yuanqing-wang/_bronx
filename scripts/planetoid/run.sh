for hidden_features in 16 # 32 64
do
    for learning_rate in 1e-2
    do
        for neighborhood_size in 1
        do
            for n_samples in 1 # 2 4 8
            do
                bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 0:10 -n 1\
                python run.py --hidden_features $hidden_features --learning_rate $learning_rate --neighborhood_size $neighborhood_size --n_samples $n_samples
            done
        done
    done
done
