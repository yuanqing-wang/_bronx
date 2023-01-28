for width in 32 64 128; do
for depth in 2 3 4 5 6; do


bsub -q gpuqueue  -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=10] span[ptile=1]" -W 0:59 -n 1 \
    python run.py --width $width --depth $depth

done; done


