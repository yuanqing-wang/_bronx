#BSUB -q cpuqueue
#BSUB -o %J.stdout
#BSUB -R "rusage[mem=5] span[hosts=1]"
#BSUB -W 2:00
#BSUB -n 64

python tune.py


