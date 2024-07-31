#!/bin/bash

# See https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0041998 for slurn usage details

# Iterate over each folder and submit a job
#!/bin/bash
#SBATCH -JSlurmProcessVideo
#SBATCH --account=gts-coc-coda20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=athomas314@gatech.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:2
#SBATCH -qinferno
#SBATCH --output=Report_%A-%a.out
#SBATCH --time=24:00:00

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# Conda initialization
__conda_setup="\$('/usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ \$? -eq 0 ]; then
    eval "\$__conda_setup"
else
    if [ -f "/usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh" ]; then
        . "/usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/bin:\$PATH"
    fi
fi
unset __conda_setup

# Activate the conda environment
conda activate /storage/home/hcoda1/0/athomas314/scratch/DEEPLABCUT

export JUST_IMAGE=1
export BATCHSIZE=9
export EPOCHS=60
export MODEL='VIT_H_14'

# Call the python script with the specified folder
python /storage/home/hcoda1/0/athomas314/ondemand/cichlid-behavior-detection/behavior_detection/transformer_training/runner_for_image_input.py 