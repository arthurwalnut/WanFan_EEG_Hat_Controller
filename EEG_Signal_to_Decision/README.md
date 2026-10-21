# EEG Hat 解码模块

本目录提供当前实验台使用的三套算法，保留 `bci` 模块路径以兼容已保存的模型。

| 模块 | 功能 |
|---|---|
| `src/bci/hat_pipeline` | 运动想象：连续预处理、1.5 秒窗口 / 0.1 秒步长、CSP / 频带功率、LDA / SVM |
| `src/bci/cvep_pipeline` | c-VEP：编码约定、试次检查、个人模板校准与匹配 |
| `src/bci/ssvep_pipeline` | SSVEP：滤波器组 CCA、留出评估与拒识阈值校准 |

网页服务直接调用这些模块，安装和操作见 [项目说明](../README.md)。`pyproject.toml` 描述算法包；依赖版本记录在 `EEG_Realtime/requirements-tested.txt`。

## 命令行训练运动想象模型

先按根目录说明配置 EEG_Realtime 的 Python 环境，在本目录运行：

```powershell
.\run_hat.ps1 train --recordings ..\EEG_Realtime\data\mi_recordings\mi_实际目录 --output .\exp_results\model.joblib --feature combined --classifier svm
```

`--recordings` 后可接多段完整录制目录；`--feature` 可选 `bandpower`、`csp`、`combined`，`--classifier` 可选 `lda`、`svm`。输出路径必须未被使用。

此入口生成模型及结果摘要，供脚本使用；网页训练会自动登记模型列表。数据筛选、划分和质量规则见 [运动想象工作流](../EEG_Realtime/docs/MI_WORKFLOW.md)。
