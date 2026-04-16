#!/bin/bash
#SBATCH --job-name=bert-augmented
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --nv \
  --bind /share/apps:/share/apps \
  --overlay /scratch/vc2836/my_overlay.ext3:ro \
  /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c "
    cd /scratch/vc2836/part1
    /share/apps/pyenv/py3.9/bin/python main.py --train_augmented --eval_transformed
  "
