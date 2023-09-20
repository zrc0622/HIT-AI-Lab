## 零样本图片分类
1. 参照 [github文档](https://github.com/OFA-Sys/Chinese-CLIP#%E9%9B%B6%E6%A0%B7%E6%9C%AC%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB) 对数据进行处理
2. 执行以下命令
```
bash run_scripts\zeroshot_eval.sh 0 E:\Data flowers ViT-B-16 RoBERTa-wwm-ext-base-chinese E:\Data\pretrained_weights\clip_cn_vit-b-16_finetune_cifar-100\clip_cn_vit-b-16_finetune_cifar-100.pt
```        