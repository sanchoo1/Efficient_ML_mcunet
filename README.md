# Efficient_ML_MCUNet

This project trains various **MCUNet** models on **Tiny ImageNet** and **CIFAR-100** datasets, optimized for **A100 GPUs** via **Chameleon Cloud**.

---

## Environment

- **Docker Image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- **CUDA**: 12.4
- **cuDNN**: 9
- **PyTorch**: 2.5.1

---

## Complete Workflow

```bash
# 1. Launch Docker container
docker run -it --rm --gpus all --ipc host -v /home/cc/llm-chi/torch:/workspace pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel bash

# 2. Install essential tools
apt update && apt install -y git wget unzip

# 3. Clone the repository
cd /workspace
git clone https://github.com/sanchoo1/Efficient_ML_mcunet.git
cd Efficient_ML_mcunet

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Prepare Tiny ImageNet dataset
rm -rf /workspace/Efficient_ML_mcunet/data/tiny-imagenet*
cd data
wget -nv http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip
mv tiny-imagenet-200 tiny-imagenet

# 6. Reconstruct validation set directory
cd tiny-imagenet/val
rm -rf images
mkdir images
unzip -q ../../tiny-imagenet-200.zip tiny-imagenet-200/val/images/* -d ../../
mv ../../tiny-imagenet-200/val/images/* ./images/

# 7. Organize validation images by class
awk '{print $2}' val_annotations.txt | sort -u | xargs -I{} mkdir -p images/{}
awk '{print $1 "\t" $2}' val_annotations.txt | while IFS=$'\t' read img cls; do mv images/$img images/$cls/; done

# 8. Confirm class count (should be 200)
ls images | wc -l

# 9. Train model (Tiny ImageNet, MCUNet, 1 GPU)
CUDA_VISIBLE_DEVICES=0 python train_mcunet.py --batch_size 512 --lr_finetune 0.1 --weight_decay 5e-4 --phase2_epochs 7 --num_workers 32

# 10. Manage results
mv logs/*/*.pth ./

# 11. Git operations (assuming branch train26)
git checkout -b train26
echo "data/" >> .gitignore
git add .
git commit -m "Add training results for train26"
git push origin train26
