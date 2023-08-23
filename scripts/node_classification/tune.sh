#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=5/task] span[ptile=1]"
#BSUB -W 0:59
#BSUB -n 8

bash ray_launch_cluster.sh -c "python tune.py" -n "bronx" -m 20000000000

