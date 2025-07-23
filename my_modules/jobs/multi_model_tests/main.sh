#!/bin/bash

#SBATCH --job-name=multi_model_tests_with_all
#SBATCH --partition=agpu06
#SBATCH --output=nsclc_main.txt
#SBATCH --error=nsclc_main.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmp002@uark.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --qos=gpu

export OMP_NUM_THREADS=1

# load required module
module purge
module load python/anaconda-3.14

# Activate venv
conda activate /home/nmp002/.conda/envs/dl_env
echo $SLURM_JOB_ID

cd $SLURM_SUBMIT_DIR || exit
# input files needed for job
files=/home/nmp002/data/

echo "Copying files..."
mkdir /scratch/$SLURM_JOB_ID
rsync -avq $files /scratch/$SLURM_JOB_ID
rsync -avq $SLURM_SUBMIT_DIR/*.py /scratch/$SLURM_JOB_ID
rsync -avq /home/nmp002/NSCLC/my_modules /scratch/$SLURM_JOB_ID
wait

cd /scratch/$SLURM_JOB_ID/ || exit

echo "Python script initiating..."
python3 multi_model_tests.py

rsync -av -q /scratch/$SLURM_JOB_ID/ $SLURM_SUBMIT_DIR/

# check if rsync succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync files back to original directory. Check /scratch/$SLURM_JOB_ID/ for output files."
  exit 1
fi
