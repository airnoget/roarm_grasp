# 当前版本快速开始（2026-11-15）

这是 RoArm-M3-S + 末端 Gemini 335L 的台架抓取工程。当前实机路径是 **YOLO RGB 检测 + 深度几何定位 + 手眼变换 + 关节反馈执行**。GG-CNN相关脚本保留作实验，不是当前默认抓取后端。

## 获取与安装

下载仓库 ZIP 后解压，或用 Git 克隆。建议 Windows x64、Python 3.12；串口默认 COM6。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-vision.txt
git clone https://github.com/waveshareteam/roarm_ws.git .vendor/roarm_ws
git -C .vendor/roarm_ws checkout 40dbd84b553695212fab713e8465f817ba95454d
```

Gemini 335L 还需要与本机 Python、相机固件兼容的 Orbbec Python SDK（模块名 `pyorbbecsdk`），按 [Orbbec 官方 SDK](https://github.com/orbbec/pyorbbecsdk) 的安装说明操作。本仓库不打包原电脑的 Python、CUDA、SDK安装缓存。GPU推理需安装与显卡环境匹配的 PyTorch；也可先用CPU验证。

## 包含什么

- `weights/targets-best.pt`：本工程训练的单类 `artificial_object` YOLO权重，识别五种台架物块。只负责检测，不是GG-CNN权重。
- `camera_profiles/`：冻结的相机预设、速度参数、夹爪配置及近似TCP。
- `.vendor/wifi-debug/RoArm-M3_example/`：修改后的ESP32控制器固件源码。
- `firmware_ui/`：网页源码和生成脚本；`releases/`附带v11应用区固件。
- `calibration/active-handeye.json`及其引用报告：原台架的外参示例，用于解释坐标链。

原始RGB-D数据集、人物画面、训练过程、串口日志、编译缓存和本机依赖未上传。`trained_model.py`在没有原训练目录时自动加载附带的YOLO权重。

## 使用

先只看图像：

```powershell
python debug.py geom
```

连接自己的相机/支架后，应先完成自己的手眼标定及TCP核对。附带外参、TCP和临时偏置是原台架配置，不能作为另一台机械臂的已验证几何参数。相机预设原始方向为0、显示旋转180度。

原台架当前TCP为link5局部 `[0,0,90]` mm，临时向画面左补20mm、额外下降10mm；这些是调试值。正式TCP标定后必须清除临时偏置。

当前Gemini固定支架下，张开的绿色夹爪进入画面角落。`camera_profiles/gripper-exclusion.json`保存原始RGB坐标中的两个固定排除多边形；在旋转显示前过滤夹爪检测框，并从几何支撑面采样中去除对应像素，不修改保存的RGB-D、不按绿色全局过滤。窗口显示`gripper ignored`计数。该配置仅适用于当前相机支架和1280×720原始RGB；分辨率不符时不应用。换支架后需重新检查区域。

在自己的定位和动作检查完成后进入控制窗口：

```powershell
python debug.py approach
```

| 按键 | 功能 |
| --- | --- |
| P / G / D | 锁定规划 / 到悬停位 / 下降 |
| C / O | 闭爪到记录角 / 张开 |
| U | 收回肩肘腕，底座角保持 |
| B | 收回后执行放块关键帧、释放，并按反序返回起点；未配置路径时保留原转180度方式 |
| N | 逐点执行已配置的放块路径，每点暂停，末点用O手动释放 |
| K | 松爪到位后，沿记录点逐点反向返回，每点暂停 |
| H | 先收回，再将底座回初始方向 |
| A | 自动接近、下降、闭爪、收回、关键帧放料及反向返程；仍在台架调试 |
| W | 记录实测候选关键帧；新记录不会自动加入执行路径 |
| X | 取消队列并保持当前位置 |
| Q | 保持并退出 |

底座为有边界的位置控制，总范围约359.8度；不支持无限连续转圈。速度按远近和动作阶段调整。初始关节角为底座0.44、肩0、肘160、腕俯仰-38、腕旋转3度，夹爪参考开口295度、闭合335度。

## 固件

v11页面版本为 `wifi-debug-20261115-v11-base360-pitch38`。附带 `.bin` 是原ESP32台架的 **0x10000应用区镜像**，不是任意板型可用的完整Flash镜像。构建、库及板型配置见 [固件说明](firmware_ui/FIRMWARE.md)。

## 验证与边界

```powershell
python -m unittest test_live_approach test_rear_transfer test_motion_speeds test_tcp_calibrate
```

这些测试验证软件逻辑和合成几何；不能证明其他实机无碰撞、TCP准确或抓取可靠。当前多个脚本保留历史实验入口，具体数据和参数说明见原README及相关文档。
# 探索与坐标输出（2026-11-16）

在工程根目录运行 `python explore_demo.py` 可离线演示；实机沿用 `python debug.py approach`，多目标默认选择最近目标，Tab切换。选中目标的二维相对坐标输出至 `exploration_output/relative-target.json`（米、arm_base参考系），Z留在抓取工程。详见 [接口说明](docs/RELATIVE_XY_TO_WORLD.md) 与 [探索方案](docs/EXPLORATION_GRASP_PLAN.md)。底盘通信和扫描示教接口待接入，实时目标编号当前适用固定底盘。

