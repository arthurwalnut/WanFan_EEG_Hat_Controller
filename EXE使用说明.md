# EEG Hat EXE 使用说明

`WanFanEEGHatController.exe` 是 Windows x64 单文件版本，内含 Python 和运行依赖，无需安装 Python。保留运动想象、c-VEP、SSVEP、通用二分类和 EEG 侧机械臂指令发送功能。

## 启动与退出

1. 将 EXE 放在可写的本地文件夹，双击运行。浏览器默认打开 `http://127.0.0.1:8765/`。首次启动需要解包依赖。
2. 保留程序窗口。结束时先停止录制和机械臂发送，等待训练结束，再在程序窗口按 Ctrl+C。正常退出会等待后台保存；直接关闭窗口或强制结束进程会中断这一流程。
3. 如果 8765 已有 EEG 服务，本次只打开已有页面，不启动第二个服务；原后台服务继续运行。

四个入口分别为 `/`、`/cvep`、`/ssvep`、`/legacy`，操作见 [项目 README](README.md)。启动不会自动连接帽子、开始刺激或发送动作。

在 PowerShell 中可选择其他端口或仅检查环境：

```powershell
.\WanFanEEGHatController.exe --port 8766
.\WanFanEEGHatController.exe --check
```

## 数据与已有模型

数据保存在 **EXE 同级的 `EEG_Realtime/data/`**，不会写入临时解包目录。

- 在当前 `EEG_Hat` 根目录运行，使用现有录制和模型，包括尾号 `b8f984` 的模型。
- 复制到另一台电脑时，EXE 本身不包含个人数据。需要历史数据时，将整个 `EEG_Realtime/data/` 按相同相对位置一起复制。
- 替换 EXE 升级时保留 `data/`。原始录制需保留完整目录，不能只复制 CSV。
- 通用工作台仍可读取同级 `InBrain/recording/`；厂商软件与其文件未嵌入 EXE。

指定其他工作区：

```powershell
.\WanFanEEGHatController.exe --workspace 'D:\EEG实验' --port 8766
```

该命令写入 `D:\EEG实验\EEG_Realtime\data`。指定工作区时，端口被占用会报错，不会改用已有服务。不同实例不要同时连接同一帽子或写入同一数据目录。

## 设备联网

帽子与电脑需在同一局域网，发现后连接。新 EXE 的防火墙程序路径与旧 Python 不同，原 Python 规则不会自动应用到 EXE；打包过程没有更改本机防火墙。

采集需接收设备源端口 **UDP 3333** 发往本机 **UDP 3333、3334** 的数据，控制连接使用设备 **TCP 3333**。为实际 EXE 路径和所用局域网配置规则。

仅当使用 Windows 默认热点 `192.168.137.1/24` 时，可在 EXE 所在目录用**管理员 PowerShell**运行：

```powershell
$eegExe = (Resolve-Path -LiteralPath '.\WanFanEEGHatController.exe').Path
New-NetFirewallRule -Name 'EEG-Hat-EXE-Hotspot-UDP' -DisplayName 'EEG Hat EXE hotspot UDP' `
  -Direction Inbound -Action Allow -Enabled True -Profile Public,Private `
  -Program $eegExe -Protocol UDP -LocalAddress 192.168.137.1 -LocalPort 3333,3334 `
  -RemoteAddress 192.168.137.0/24 -RemotePort 3333
```

如该 EXE 已有显式阻止规则，需在“高级安全 Windows Defender 防火墙”中核对该程序的阻止项；新增允许规则不能覆盖阻止规则。移动 EXE 后需更新程序路径。移除此新增规则用 `Remove-NetFirewallRule -Name 'EEG-Hat-EXE-Hotspot-UDP'`。

## 机械臂电脑

EXE 运行在 EEG 电脑。Linux 机械臂电脑继续使用 [robot_pc](robot_pc/README.md) 中的接收器与动作脚本，运行步骤不变。只在 EEG 电脑查看分类时，无需启动机械臂接收器。

## 从源码重新构建

构建需要 Windows x64、Python 3.12。新环境先安装两个固定版本清单：

```powershell
py -3.12 -m venv .build\python
.\.build\python\Scripts\python.exe -m pip install -r EEG_Realtime\requirements-tested.txt -r windows_exe\requirements-build.txt
.\.build\python\Scripts\python.exe -B windows_exe\build.py
```

输出为 `.build/output/WanFanEEGHatController.exe`。本机已有 `.packages` 时，构建器优先使用它并核对版本；源码打包范围由 `windows_exe/EEG_Hat.spec` 明确列出，不包含录制、模型、厂商应用或机器配置。

资源与持久数据分开定位的依据见 [PyInstaller 运行时说明](https://pyinstaller.org/en/stable/runtime-information.html)。
