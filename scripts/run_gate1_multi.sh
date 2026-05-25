#!/bin/bash
set -e
cd /root/autodl-tmp/PEA-MedSeg

AMOS_IMG_DIR=/root/autodl-tmp/data/amos22/imagesTr
AMOS_LBL_DIR=/root/autodl-tmp/data/amos22/labelsTr
OUTPUT_BASE=results/vulnerability_multi

for i in 01
02
03
04
05
06
07
08
09
10; do
    CASE=amos_00
    IMG=/.nii.gz
    LBL=/.nii.gz
    
    if [ ! -f "" ]; then
        echo "SKIP:  (not found)"
        continue
    fi
    
    echo "===  ==="
    CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python test_vulnerability.py         --sam-root /root/autodl-tmp/SAM-Med3D         --checkpoint /root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth         --image ""         --label ""         --output-dir "/"         --noise-levels 0 5 10 15 20         --trials 3         --seed 42         --prompt random
done
echo "Done"
