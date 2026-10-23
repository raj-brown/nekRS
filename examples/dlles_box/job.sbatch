#!/bin/bash
#SBATCH -J NEKRS_LES
#SBATCH -N 1
#SBATCH --partition=3090-gcondo
##SBATCH -A bkeith1-gcondo
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH -o RANS_job_%j.o
#SBATCH -e RANS_job_%j.e



export NEKRS_HOME=/oscar/data/gk/kshukla1/.local/nekrs_wrap
export PATH=$NEKRS_HOME/bin:$PATH

module load hpcx-mpi/2.25.1s-le4f
module load cuda/12.9.0-cinr


ulimit -c 0
export NEKRS_CACHE_DIR=/oscar/data/gk/kshukla1/nekrs_cache/dlles_box
mkdir -p $NEKRS_CACHE_DIR


srun --mpi=pmix nekrs --setup combined.par
