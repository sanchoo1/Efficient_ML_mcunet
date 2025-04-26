# 1. clear
rm -rf /workspace/Efficient_ML_mcunet/data/tiny-imagenet*

# 2. download Tiny ImageNet
cd /workspace/Efficient_ML_mcunet/data
wget -nv http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip
mv tiny-imagenet-200 tiny-imagenet

# 3. reconstruct val
cd tiny-imagenet/val
rm -rf images
mkdir images
unzip -q ../../tiny-imagenet-200.zip tiny-imagenet-200/val/images/* -d ../../
mv ../../tiny-imagenet-200/val/images/* ./images/

# 4. 
awk '{print $2}' val_annotations.txt | sort -u | xargs -I{} mkdir -p images/{}
awk '{print $1 "\t" $2}' val_annotations.txt | while IFS=$'\t' read img cls; do mv images/$img images/$cls/; done

# 5. class counts
ls images | wc -l   # 应该是 200


run
CUDA_VISIBLE_DEVICES=0 python train_mcunet.py --batch_size 512 --lr_finetune 0.1 --weight_decay 5e-4 --phase2_epochs 7 --num_workers 32
