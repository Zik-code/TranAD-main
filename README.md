[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/TranAD/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FTranAD&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# TranAD
This repository supplements our paper "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data" accepted in VLDB 2022. This is a refactored version of the code used for results in the paper for ease of use. Follow the below steps to replicate each cell in the results table. The code is provided as-is. Due to limited resources, we are unable to provide support on any issues you may experience with installing or running the tool.

Our work has been discussed in the PodBean podcast! [See here](https://papersread.ai/e/tranad-deep-transformer-networks-for-anomaly-detection-in-multivariate-time-series-data-1663142096/). 

## Results
![Alt text](results/main.PNG?raw=true "results")

## Installation
This code needs Python-3.7 or higher.
```bash
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```

## Dataset Preprocessing
Preprocess all datasets using the command
```bash
python3 preprocess.py SMAP MSL SWaT WADI SMD MSDS UCR MBA NAB
```
Distribution rights to some datasets may not be available. Check the readme files in the `./data/` folder for more details. If you want to ignore a dataset, remove it from the above command to ensure that the preprocessing does not fail.

## Result Reproduction
To run a model on a dataset, run the following command:
```bash
python3 main.py --model <model> --dataset <dataset> --retrain
```
where `<model>` can be either of 'TranAD', 'GDN', 'MAD_GAN', 'MTAD_GAT', 'MSCRED', 'USAD', 'OmniAnomaly', 'LSTM_AD', and dataset can be one of 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'MSDS', 'MBA', 'UCR' and 'NAB. To train with 20% data, use the following command 
其中 <model> 可以是 'TranAD'、'GDN'、'MAD_GAN'、'MTAD_GAT'、'MSCREAD'、'USAD'、'OmniAnomaly'、'LSTM_AD' 中的任意一个，<dataset> 可以是 'SMAP'、'MSL'、'SWaT'、'WADI'、'SMD'、'MSDS'、'MBA'、'UCR' 和 'NAB' 中的一个。要使用 20% 的数据进行训练，请使用以下命令
```bash
python3 main.py --model <model> --dataset <dataset> --retrain --less
```
You can use the parameters in `src/params.json` to set values in `src/constants.py` for each file. 

> Note: to reproduce exact results of baselines, use their original codebases (links given in our paper) as the ones implemented in this repository are *not* the ones used in the paper, which used the original versions. The versions provided here are for use of initial comparison and may not be identical to the original versions.

For ablation studies, use the following models: 'TranAD_SelfConditioning', 'TranAD_Adversarial', 'TranAD_Transformer', 'TranAD_Basic'.

The output will provide anomaly detection and diagnosis scores and training time. For example:
```bash
$ python3 main.py --model TranAD --dataset SMAP --retrain 
Using backend: pytorch
Creating new model: TranAD
Training TranAD on SMAP
Epoch 0,        L1 = 0.09839354782306504
Epoch 1,        L1 = 0.039524692888342115
Epoch 2,        L1 = 0.022258711623482686
Epoch 3,        L1 = 0.01833707226553135
Epoch 4,        L1 = 0.016330517334598792
100%|███████████████████████████████████████████████████████████████████| 5/5 [00:03<00:00,  1.57it/s]
Training time:     3.1920 s
Testing TranAD on SMAP
{'FN': 0,
 'FP': 182,
 'Hit@100%': 1.0,
 'Hit@150%': 1.0,
 'NDCG@100%': 0.9999999999999999,
 'NDCG@150%': 0.9999999999999999,
 'TN': 7575,
 'TP': 748,
 'f1': 0.8915325929177795,
 'precision': 0.8043010666204187,
 'recall': 0.9999999866310163,
 'threshold': 0.16133320075167037}
```

All outputs can be run multiple times to ensure statistical significance. 

## Supplementary video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/b2fSzneXPsg/0.jpg)](https://www.youtube.com/watch?v=b2fSzneXPsg)

## Cite this work

Our paper is available in the Proceedings of VLDB: http://vldb.org/pvldb/vol15/p1201-tuli.pdf.
If you use this work, please cite using the following bibtex entry.
```bibtex
@article{tuli2022tranad,
  title={{TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data}},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  journal={Proceedings of VLDB},
  volume={15},
  number={6},
  pages={1201-1214},
  year={2022}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shreshth Tuli.
All rights reserved.

See License file for more details.
import pandas as pd
import numpy as np

# 人口统计学特征
age = np.random.randint(18, 70, 200)  # 年龄范围在18到70岁之间
gender = np.random.choice(['男', '女'], 200)  # 随机选择性别
regions = ['一线城市', '二线城市', '三线城市', '四线城市', '农村地区']
region = np.random.choice(regions, 200)  # 随机选择地区
family_income = np.random.randint(2000, 50000, 200)  # 家庭收入范围在2000到50000元之间
family_size = np.random.randint(1, 7, 200)  # 家庭人口数范围在1到6人之间

# 智能家居相关特征
has_smart_home = np.random.choice([0, 1], 200)  # 0表示没有，1表示有
smart_home_needs = ['安防', '节能', '便捷控制', '健康监测', '娱乐']
smart_home_need = [np.random.choice(smart_home_needs) for _ in range(200)]  # 随机选择智能家居功能需求
awareness_level = np.random.choice(['不了解', '了解一点', '比较了解', '非常了解'], 200)  # 对智能家居的了解程度
usage_frequency = np.random.choice(['很少用', '偶尔用', '经常用', '每天用'], 200)  # 智能家居使用频率

# 品牌相关特征
used_xiaomi_products = np.random.choice([0, 1], 200)  # 0表示没有使用过，1表示使用过
satisfaction_level = np.random.choice(['不满意', '一般', '满意', '非常满意'], 200)  # 对小米品牌的满意度
xiaomi_smart_home_awareness = np.random.choice(['不知道', '听说过', '了解一点', '比较了解', '非常了解'], 200)  # 对小米智能家居品牌的认知度

# 其他特征
purchase_budget = np.random.randint(500, 10000, 200)  # 购买预算范围在500到10000元之间
purchase_channel = np.random.choice(['线上', '线下'], 200)  # 购买渠道偏好
promotion_sensitivity = np.random.choice(['不敏感', '一般', '敏感'], 200)  # 促销活动敏感度

# 目标变量：是否购买（随机生成，这里只是简单模拟，实际可根据更复杂的逻辑生成）
purchase = np.random.choice([0, 1], 200)

# 创建DataFrame
data = {
    '年龄': age,
    '性别': gender,
    '地区': region,
    '家庭收入': family_income,
    '家庭人口数': family_size,
    '是否已有智能家居设备': has_smart_home,
    '对智能家居功能的需求': smart_home_need,
    '对智能家居的了解程度': awareness_level,
    '智能家居使用频率': usage_frequency,
    '是否使用过小米其他产品': used_xiaomi_products,
    '对小米品牌的满意度': satisfaction_level,
    '对小米智能家居品牌的认知度': xiaomi_smart_home_awareness,
    '购买预算': purchase_budget,
    '购买渠道偏好': purchase_channel,
    '促销活动敏感度': promotion_sensitivity,
    '是否购买': purchase
}

df = pd.DataFrame(data)

print(df.head(100))