#!/bin/bash
#SBATCH --ntasks=64
#SBATCH --job-name=AlBulk2
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-user=alexandre.cloots@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00
export OMP_NUM_THREADS=1
unset SLURM_CPUS_PER_TASK
mpirun -np 1 abinit < File_Bulk2Atoms.files >& log
echo "--"
