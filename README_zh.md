# STMP 定位实验复现代码

对应论文 **Enhancing Localization Accuracy via Score-Based Interference Suppression**，作者 Boyu Teng、Xiaojun Yuan。[论文 DOI](https://doi.org/10.1109/LWC.2026.3708142) · [English](README.md) · [论文引用](#论文引用)

项目自有代码使用 **MIT**，版权署名 **Boyu Teng**。原代码中的作者署名保留。QuaDRiGa、Score-SDE、EMA 和 NVIDIA 相关代码各自保留原许可证；其中 QuaDRiGa 与 NVIDIA 组件含非商业条款，整个依赖包不能视为全部使用 MIT。详见 [第三方许可说明](THIRD_PARTY_NOTICES.md)。`baseline/` 中是 Boyu Teng 提供的 MDVALSE 非官方实验实现，并非参考论文作者的官方仓库发布版。

## 包含内容

本包包括信道生成、两阶网络训练、冻结模型、共享观测生成、STMP/ML/CRB 仿真、MDVALSE 基线、E0–E3 绘图以及数值与接口检查。正文对应的主要实验及 A/B 迁移场景均有入口。

为了控制大小，只提供已完成复现的结果图、汇总 CSV 和模型选择元信息。训练数据、模型权重、逐样本结果 MAT、缓存及机器日志不随包分发。从零复现会在 `work/` 下生成这些文件，不需要旧 checkpoint。结果文件是本次重新生成数据、重新训练后的复现结果，不是论文原始 MAT。

## 先查看结果

- [C 场景性能曲线](results/reference/figures/fresh_historical/E1_all_trials.png)
- [迭代性能曲线](results/reference/figures/fresh_historical/E2_all_trials.png)
- [A/B 泛化曲线，含两条 MDVALSE 基线](results/reference/figures/fresh_historical/E3_all_trials.png)
- [训练验证曲线](results/reference/figures/training_validation.png)

无需 MATLAB、GPU、MAT 或权重即可用汇总 CSV 重画图：

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_reference_results.py --output work/reference_plots
```

这一步只重画已提供的参考结果，不是重新仿真。

## 从零复现的顺序

完整可执行命令及环境配置见 [REPRODUCING.md](docs/REPRODUCING.md)。在发布目录中运行，或向 Python 传入 `--root`、向 MATLAB 传入同一实验输出目录。

1. MATLAB：`addpath('matlab'); run_channel_generation(fullfile(pwd,'work'),8,'full')`。
2. Python：`python scripts/run_experiment.py observations`。
3. 一阶：`python scripts/run_experiment.py train1`，检查 C validation 后运行 `freeze1`。
4. 二阶：`python scripts/run_experiment.py train2`，检查 C validation 后运行 `freeze2`。教师模型始终冻结。
5. Python：依次运行 `covariance`、`evaluate`。
6. MATLAB：`run_baseline(fullfile(pwd,'work'),8,3000)`，对 C/A/B 的同一观测运行基线。
7. Python：依次运行 `plots`、`e0`。

训练设有最大步数和累计时间限制，可以在冻结前用 `--resume` 续训。只能用 C 验证集选择模型；C 测试集与 A/B 场景不用于选择。冻结后禁止在原训练目录中继续改写模型。

本地 MATLAB 与服务器 Python 分开运行时，复制 `work/datasets/`、`work/observations/` 和最终结果目录，并校验文件 SHA-256。发布代码不包含服务器地址、用户名或自动 SSH 操作。完整训练参考环境为 Linux、Python 3.11、PyTorch 2.4.1、A800 80 GB；本地已提供 CPU 小样本验证流程。运行 GPU 作业前自行查看 `nvidia-smi` 并设置 `CUDA_VISIBLE_DEVICES`。

## 如何解读结果

默认 `historical` 沿用已核对的实际程序数值约定；`paper` 是同时改变初始化及方差约定的单独诊断。主图使用全样本统计，另给出历史筛样图，二者不能混同。

E2 按要求只显示 kappa=0、5、10 dB；-5 dB 原始实验仍运行，汇总表仍保留。强干扰坏点没有被删除或伪装成成功样本。A/B 图已包含 MDVALSE。收敛、筛样、非有限结果及批处理差异详见 [RESULTS.md](docs/RESULTS.md)。

许可证、引用文件、检查脚本及源码哈希清单都在包内。`docs/VERIFICATION.md` 区分此次发布验证和此前完整复现；整理代码时没有再次进行完整训练。

## 论文引用

如果本代码或实验结果对您的研究有帮助，请引用对应的原始论文：

Boyu Teng, Xiaojun Yuan, “Enhancing Localization Accuracy via Score-Based Interference Suppression,” *IEEE Wireless Communications Letters*, vol. 15, pp. 3836–3840, 2026. [DOI: 10.1109/LWC.2026.3708142](https://doi.org/10.1109/LWC.2026.3708142).

[IEEE Xplore 原文](https://ieeexplore.ieee.org/abstract/document/11586053) · [下载 BibTeX](CITATION.bib) · [GitHub 引用元数据](CITATION.cff)

```bibtex
@article{Teng2026STMP,
  author  = {Teng, Boyu and Yuan, Xiaojun},
  title   = {{Enhancing Localization Accuracy via Score-Based Interference Suppression}},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  volume  = {15},
  pages   = {3836--3840},
  doi     = {10.1109/LWC.2026.3708142},
  url     = {https://ieeexplore.ieee.org/abstract/document/11586053}
}
```
