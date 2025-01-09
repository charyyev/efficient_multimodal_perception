#!/bin/bash

#SBATCH --ntasks=6                        
#SBATCH --time=56:00:00    
#SBATCH --mem-per-cpu=16000
#SBATCH --cpus-per-task=4
#SBATCH --tmp=500G                         
#SBATCH --job-name=surf_sam_s
#SBATCH --output=./jobs/%j.out
#SBATCH --error=./jobs/%j.err
#SBATCH --gpus=6
#SBATCH --gres=gpumem:23g


source ~/miniconda3/etc/profile.d/conda.sh
conda activate unimae

rsync -q /cluster/scratch/scharyyev/thesis/nuscenes-sam.tar.gz ${TMPDIR}
cd ${TMPDIR}
tar --use-compress-program=pigz -xf nuscenes-sam.tar.gz
cd /cluster/home/scharyyev/proj/UniM2AE/Pretrain/


#module load gcc/8.2.0

srun --gres=gpumem:23g python -u tools/train.py \
                                 configs/triplane_occ2.py \
                                 --work-dir /cluster/scratch/scharyyev/thesis/models/finetune/triplane_range/surf_sam_occ_s  \
                                 --launcher="slurm" \
                                 --data-root ${TMPDIR}/nuscenes-full