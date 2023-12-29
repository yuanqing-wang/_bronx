#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=4:j_exclusive=yes"
#BSUB -R "rusage[mem=5/task] span[ptile=1]"
#BSUB -W 1:59
#BSUB -n 4

bash ray_launch_cluster.sh -c "python tune.py" -n "bronx" -m 20000000000

