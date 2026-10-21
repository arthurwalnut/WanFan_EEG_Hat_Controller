# EEG 实验服务

安装、连接设备和网页操作见根目录的 [项目说明](../README.md)。

本目录包含服务端 `eeg_realtime`、四套网页、依赖清单及本机 `data`。默认入口为 `http://127.0.0.1:8765/`；`/cvep`、`/ssvep`、`/legacy` 提供其他现有工作台。仅一套工作台可占用采集资源。

## 启动

配置依赖后双击 `启动.bat`，或在本目录运行：

```powershell
.\run.ps1 -NoBrowser
```

端口可用 `-Port 8767` 指定。手动启动时保持终端运行，用 Ctrl+C 结束。

存在 `.packages/runtime.json` 时，启动器固定使用其中的 Python，与本机防火墙规则一致；否则查找 `.venv` 等可用环境。换电脑请重新建立虚拟环境，不复制 `.packages` 或旧机器配置。

## 本机后台服务与网络维护

当前开发电脑使用 `tools/service_watchdog.py` 和登录启动项保持服务可用。查询状态：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\configure-autostart.ps1 -Status
```

同一脚本的 `-Enable` / `-Disable` 管理此电脑的登录启动项；取消自启不终止已运行的服务。日志位于 `.logs/`，勿删除正在使用的守护锁。服务恢复后保持空闲，需手动重新开始采集。

本机热点范围规则由 `tools/configure-hotspot-firewall.ps1` 管理：`-Status` 查询，`-Install` 安装，`-Restore` 恢复。更改需管理员权限。规则限定已配置 Python、热点 `192.168.137.0/24`、设备源端口 UDP 3333、本机 `192.168.137.1` 的 UDP 3333/3334；备份保存在 `data/network/`。

这些工具依赖当前电脑的 Python 路径与历史配置，保留用于本机维护，默认不纳入公开源码。其他电脑按根目录网络说明配置。

## 详细说明

- [运动想象录制、训练、回放与机械臂](docs/MI_WORKFLOW.md)
- [c-VEP 原理与操作](docs/CVEP_GUIDE.md)
- [c-VEP 中断排查](docs/CVEP_TROUBLESHOOTING.md)
- [SSVEP 操作](docs/SSVEP_GUIDE.md)
- [EEG Hat 通信协议](docs/PROTOCOL.md)

录制、模型与配置统一保存在 `data/`；原始数据不要只复制 CSV。已有模型的反序列化依赖同级 `EEG_Signal_to_Decision/src/bci`，请保留目录相对位置。
