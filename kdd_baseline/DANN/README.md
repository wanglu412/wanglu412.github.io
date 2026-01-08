# DANN - Domain-Adversarial Training of Neural Networks

基于论文 Ganin et al. (2015) 的完整实现，用于药物发现中的域适应任务。

## 📁 文件结构

```
DANN/
├── dann_model.py           # 模型架构（GRL、特征提取器、分类器）
├── dann_trainer.py         # 训练和评估逻辑
├── dann_data_loader.py     # 数据加载工具
├── train_dann.py          # 主训练脚本（单数据集）
├── train_all_datasets.py  # 批量训练脚本（多数据集顺序训练）
├── run_5times.bat         # Windows批处理：运行命令5次
├── run_5times.sh          # Shell脚本：运行命令5次
├── run_multiple.py        # Python脚本：运行命令多次（可指定次数）
├── test_dann.py           # 单元测试
├── list_datasets.py       # 列出所有可用数据集
├── dann_requirements.txt  # 依赖包
└── README.md             # 本文件
```

## 🎯 核心特性

1. **梯度反转层 (GRL)**: DANN的核心机制
2. **图神经网络**: 支持GIN和GCN
3. **动态Lambda调度**: 按照论文实现
4. **AUC评估**: 分子性质预测的主要指标
5. **单数据集训练**: 每次训练一个数据集，结果清晰
6. **18种数据集配置**: 全面评估

## 📊 支持的数据集

### GOOD数据集 (12种配置)

**GOODHIV** - HIV活性预测 (4种):
- `goodhiv_scaffold_covariate` - 骨架域，协变量偏移
- `goodhiv_scaffold_concept` - 骨架域，概念偏移
- `goodhiv_size_covariate` - 大小域，协变量偏移
- `goodhiv_size_concept` - 大小域，概念偏移

**GOODZINC** - 分子性质预测 (4种):
- `goodzinc_scaffold_covariate`
- `goodzinc_scaffold_concept`
- `goodzinc_size_covariate`
- `goodzinc_size_concept`

**GOODPCBA** - 生物活性预测 (4种):
- `goodpcba_scaffold_covariate`
- `goodpcba_scaffold_concept`
- `goodpcba_size_covariate`
- `goodpcba_size_concept`

### DrugOOD数据集 (6种配置)

**IC50** (3种):
- `ic50_assay` - 基于assay分割
- `ic50_scaffold` - 基于骨架分割
- `ic50_size` - 基于大小分割

**EC50** (3种):
- `ec50_assay`, `ec50_scaffold`, `ec50_size`

## 🚀 快速开始

### 0. 数据准备

**数据会自动从上级目录的 `data/` 文件夹加载，无需手动下载！**

确保数据在正确的位置：
```
KDD_baseline/
├── DANN/          # ← 你在这里
└── data/          # ← 数据在这里（自动加载）
    ├── GOODHIV/
    ├── GOODZINC/
    ├── GOODPCBA/
    ├── lbap-ic50_assay-chembl30/
    ├── lbap-ic50_scaffold-chembl30/
    ├── lbap-ic50_size-chembl30/
    ├── lbap-ec50_assay-chembl30/
    ├── lbap-ec50_scaffold-chembl30/
    └── lbap-ec50_size-chembl30/
```

代码会自动从 `../data/` 读取数据，不需要任何配置！

### 1. 安装依赖

```bash
pip install -r DANN/dann_requirements.txt
```

### 2. 查看可用数据集

```bash
cd DANN
python list_datasets.py
```

### 3. 训练单个数据集

```bash
# 训练默认数据集（GOODHIV scaffold covariate）
python train_dann.py

# 训练指定数据集
python train_dann.py --dataset goodhiv_scaffold_covariate

# 快速测试（10个epoch）
python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10

# 自定义参数
python train_dann.py \
    --dataset goodhiv_scaffold_covariate \
    --batch_size 64 \
    --epochs 200 \
    --hidden_dim 256 \
    --num_layers 4
```

### 4. 批量训练多个数据集（顺序执行）

```bash
# 训练所有GOOD数据集（12个，依次执行）
python train_all_datasets.py good

# 训练所有IC50数据集（3个，依次执行）
python train_all_datasets.py ic50

# 训练所有EC50数据集（3个，依次执行）
python train_all_datasets.py ec50

# 训练所有数据集（18个，依次执行，需6-12小时）
python train_all_datasets.py all
```

### 5. 重复运行同一命令（用于多次实验）

我们提供了三种脚本来自动运行命令多次，并自动计算Test AUC的平均值和标准差：

**Windows批处理文件：**
```bash
# 运行命令5次，自动计算统计结果
run_5times.bat "python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10"
```

**Shell脚本（Linux/Mac/Git Bash）：**
```bash
# 先添加执行权限
chmod +x run_5times.sh

# 运行命令5次，自动计算统计结果
./run_5times.sh "python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10"
```

**Python脚本（推荐，功能最强）：**
```bash
# 默认运行5次
python run_multiple.py "python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10"

# 自定义次数（比如3次）
python run_multiple.py 3 "python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10"
```

这些脚本会：
- 自动运行命令指定次数
- 显示当前是第几次运行
- 记录每次运行的成功/失败状态
- **自动提取每次运行的Test AUC值**
- **计算Test AUC的平均值、标准差、最小值、最大值**
- **生成统计报告文件：`statistics_{dataset_name}.json`**
- 在最后显示汇总结果（格式：Mean ± Std）

**输出示例：**
```
================================================================================
TEST AUC STATISTICS
================================================================================
Dataset: goodhiv_scaffold_covariate
Successful runs with AUC: 5

  Mean AUC:  0.7623
  Std AUC:   0.0145
  Min AUC:   0.7456
  Max AUC:   0.7812

  Result: 0.7623 ± 0.0145
================================================================================
```

## ⚙️ 命令行参数

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 要训练的数据集（单个） | goodhiv_scaffold_covariate |
| `--epochs` | 训练轮数 | 100 |
| `--batch_size` | 批次大小 | 32 |
| `--lr` | 学习率 | 0.001 |
| `--hidden_dim` | 隐藏层维度 | 128 |
| `--num_layers` | GNN层数 | 3 |
| `--gnn_type` | GNN类型 (gin/gcn) | gin |
| `--patience` | 早停耐心值 | 20 |
| `--device` | 设备 (cuda/cpu) | cuda |

完整参数列表：
```bash
python train_dann.py --help
```

## 📈 输出示例

### 单数据集训练
```
================================================================================
Training Complete - GOODHIV_Scaffold_Covariate
================================================================================
Best Validation AUC: 0.7845
Test AUC: 0.7623
Test Accuracy: 0.7234
Training Time: 25.43 minutes
================================================================================
```

### 批量训练
```
================================================================================
FINAL SUMMARY - Sequential Training on Multiple Datasets
================================================================================
Dataset                        Val AUC      Test AUC     Test Acc
--------------------------------------------------------------------------------
GOODHIV_Scaffold_Covariate     0.7845       0.7623       0.7234
GOODHIV_Scaffold_Concept       0.7912       0.7734       0.7456
GOODHIV_Size_Covariate         0.8045       0.7923       0.7623
GOODHIV_Size_Concept           0.8123       0.8001       0.7712
================================================================================
```

结果保存：
- `dann_checkpoints/` - 模型文件
- `dann_results.json` - 训练结果

## 🔬 算法原理

### 模型架构
```
输入图数据
    ↓
特征提取器 (GNN)
    ↓
    ├─→ 标签分类器 → 类别预测
    └─→ [GRL] → 域分类器 → 域预测
```

### 梯度反转层 (GRL)
- **前向**: 恒等变换
- **反向**: 梯度乘以 -λ

Lambda调度：
```python
λ(p) = 2 / (1 + exp(-10p)) - 1
```
其中 p ∈ [0,1] 表示训练进度

### 损失函数
```
L_total = L_class + λ * L_domain
```

## 💡 重要说明

**一个数据集，一个模型**:
- 每次训练只使用**一个**数据集配置
- 不支持在同一次训练中混合多个数据集
- 每个数据集有独立的train/val/test分割
- 结果清晰、易于理解和比较

要在多个数据集上评估，使用批量训练脚本顺序训练多个模型。

## 🔧 故障排除

### CUDA内存不足
```bash
python train_dann.py --dataset goodhiv_scaffold_covariate --batch_size 16 --hidden_dim 64
```

### 训练速度慢
```bash
python train_dann.py --dataset goodhiv_scaffold_covariate --num_workers 4
```

### 性能不佳
调整超参数：
- 增加模型容量: `--hidden_dim 256 --num_layers 5`
- 调整域权重: `--domain_weight 0.5` 或 `--domain_weight 2.0`
- 修改学习率: `--lr 0.0001` 或 `--lr 0.01`

## 📚 引用

如果使用本实现，请引用原始论文：

```bibtex
@article{ganin2016domain,
  title={Domain-adversarial training of neural networks},
  author={Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and
          Germain, Pascal and Larochelle, Hugo and Laviolette, Fran{\c{c}}ois and
          Marchand, Mario and Lempitsky, Victor},
  journal={The journal of machine learning research},
  volume={17},
  number={1},
  pages={2096--2030},
  year={2016},
  publisher={JMLR.org}
}
```

## 📝 域和偏移类型说明

### Domain（域）类型
- **scaffold**: 按分子骨架分组，测试结构泛化能力
- **size**: 按分子大小分组，测试尺寸泛化能力

### Shift（偏移）类型
- **covariate**: 特征分布变化，P(X)变化但P(Y|X)不变
- **concept**: 标签关系变化，P(Y|X)变化（更具挑战性）

## 🎓 使用流程推荐

1. **快速测试** (2-5分钟)
```bash
python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10
```

2. **单数据集完整训练** (20-40分钟)
```bash
python train_dann.py --dataset goodhiv_scaffold_covariate
```

3. **数据集全配置评估** (1-3小时)
```bash
python train_all_datasets.py good  # 训练所有GOOD配置
```

4. **综合评估** (6-12小时)
```bash
python train_all_datasets.py all  # 训练所有18个配置
```

---

**注意**: 确保数据文件在 `../data/` 目录中。运行前从 `DANN/` 目录执行命令。
