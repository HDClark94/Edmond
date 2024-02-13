#!/bin/sh
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=10G
#$ -pe sharedmem 1

#load the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda

# activate the virtual environment
source activate myenv

# append the python path
export PYTHONPATH="/home/s1228823/Edmond"

# Run the program
python /home/s1228823/Edmond/VR_grid_analysis/FieldShuffleAnalysis/shuffle_analysis.py