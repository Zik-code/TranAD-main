
### 许可证
Python 37 38 Hits

### TranAD
本代码库是我们发表于 VLDB 2022 的论文《TranAD: 用于多元时间序列数据异常检测的深度Transformer网络》的补充材料。为便于使用，代码已重构为论文实验结果的实现版本。按以下步骤可复现结果表中的每个单元格。代码按原样提供，由于资源有限，我们无法为安装或运行工具时遇到的问题提供支持。

我们的工作已在PodBean播客中讨论！详见[此处](链接)。


### 结果
（替代文本，图片描述）


### 安装
代码需使用Python 3.7或更高版本。

```bash
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```


### 数据集预处理
使用以下命令预处理所有数据集：

```bash
python3 preprocess.py SMAP MSL SWaT WADI SMD MSDS UCR MBA NAB
```

部分数据集可能无分发权限，请查看`./data/`文件夹中的README文件了解详情。若需忽略某个数据集，从命令中移除该数据集名称以避免预处理失败。


### 结果复现
在指定数据集上运行模型：

```bash
python3 main.py --model <模型> --dataset <数据集> --retrain
```

其中`<模型>`可选值：'TranAD'、'GDN'、'MAD_GAN'、'MTAD_GAT'、'MSCRED'、'USAD'、'OmniAnomaly'、'LSTM_AD'  
`<数据集>`可选值：'SMAP'、'MSL'、'SWaT'、'WADI'、'SMD'、'MSDS'、'MBA'、'UCR'、'NAB'  

使用20%数据训练：

```bash
python3 main.py --model <模型> --dataset <数据集> --retrain --less
```

可通过`src/params.json`中的参数设置`src/constants.py`中的配置。

**注意**：若需复现基线模型的精确结果，请使用其原始代码库（论文中提供链接），因为本代码库中的实现并非论文中使用的原始版本，此处版本仅用于初步对比，可能与原始版本存在差异。

消融研究使用以下模型：'TranAD_SelfConditioning'、'TranAD_Adversarial'、'TranAD_Transformer'、'TranAD_Basic'。

输出将包含异常检测与诊断分数及训练时间。示例：

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

所有输出可多次运行以确保统计显著性。


### 补充视频
（图片替代文本，视频描述）


### 引用本文
论文收录于VLDB会议论文集：http://vldb.org/pvldb/vol15/p1201-tuli.pdf。若使用本工作，请按以下BibTeX条目引用：

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


### 许可证
BSD-3-Clause协议。版权所有 © 2022，Shreshth Tuli。保留所有权利。  
详情见许可证文件。