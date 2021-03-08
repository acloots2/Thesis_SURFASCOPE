#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=DRFAlB
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-user=alexandre.cloots@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00
#SBATCH --partition=Def
export OMP_NUM_THREADS=1
unset SLURM_CPUS_PER_TASK
source /home/ucl/naps/acloots/ABINIT-1/AlBulkSmallGrid/
mpirun -np 1 abinit < Chi0_Al_Bulk.files >& log
echo "--"
