#!/bin/bash

#SBATCH --ntasks=1                             
#SBATCH --time=2:00:00                    
#SBATCH --mem-per-cpu=16000
#SBATCH --cpus-per-task=4
#SBATCH --tmp=500G                         
#SBATCH --job-name=test_log
#SBATCH --output=./jobs/%j.out
#SBATCH --error=./jobs/%j.err
#SBATCH --gpus=1


source ~/miniconda3/etc/profile.d/conda.sh
conda activate unimae

rsync -q /cluster/scratch/scharyyev/thesis/nuscenes-full-compact.tar.gz ${TMPDIR}
cd ${TMPDIR}
tar --use-compress-program=pigz -xf nuscenes-full-compact.tar.gz
cd /cluster/home/scharyyev/proj/UniM2AE/Pretrain/


#module load gcc/8.2.0

python -u   tools/test.py \
            configs/multiscale.py \
            --checkpoint /cluster/scratch/scharyyev/thesis/rmae/range_multiscale/epoch_40.pth \
            --show-pretrain \
            --show-dir /cluster/scratch/scharyyev/thesis/unimae/vis/range_multiscale \
            --data-root ${TMPDIR}/nuscenes-full