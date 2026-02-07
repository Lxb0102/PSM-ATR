echo "mimic 3"
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203

echo "mimic 4"
python code/train.py --dataset 4 --dim 256 --batch 32 --visit 3 --seed 1203

去除预训练
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode improved
训练+保存最佳模型
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode improved --pretrain_mask --pretrain_epochs 20 --save_model 1



1）原始 backbone + 不预训练（严格 baseline，对应最早版本）
cd F:\code_2025\armr2-pretrain-main\armr2-main
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode orig
cd F:\code_2025\armr2-pretrain-main\armr2-mainpython code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode orig
2）改进 backbone + 不预训练（只看结构改进的效果）
cd F:\code_2025\armr2-pretrain-main\armr2-main
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode improved
cd F:\code_2025\armr2-pretrain-main\armr2-mainpython code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode improved
3）改进 backbone + 预训练（真正的预训练效果）
仅 Mask 预训练：
cd F:\code_2025\armr2-pretrain-main\armr2-main
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode improved --pretrain_mask --pretrain_epochs 20
cd F:\code_2025\armr2-pretrain-main\armr2-mainpython code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode improved --pretrain_mask --pretrain_epochs 20
同时 NSP + Mask 预训练：
cd F:\code_2025\armr2-pretrain-main\armr2-main
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 --backbone_mode improved --pretrain_nsp --pretrain_mask --pretrain_epochs 20


