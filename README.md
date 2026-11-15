> 当前分享版本：2026-11-15。下文部分内容是历史实验记录，当前实机快速开始见 [QUICKSTART.md](QUICKSTART.md)。
> 当前采用 YOLO + 深度几何；新夹爪已启用，TCP近似90mm；GG-CNN为可选实验路线。附带的手眼外参仅对应原台架，其他机械臂必须重新标定。

# RoArm-M3-S 腕部相机抓取

2026-11-11 实物状态：用户随后确认仅拆下夹爪开合舵机，前五关节完整，相机支架已固定；新舵机暂不安装、暂不调试，其余结构不变。原装完整夹爪仿真不能代表当前实机。当前可准备腕部相机标定，但尚无最终实测外参、工具中心与实际抓取验收。用户已打印标定板，并确认 Tag 黑色外框边长为50毫米。

比赛自动抓取使用末端彩深相机，每次触发选取一件特殊人造物。固定俯视相机和矩形后端可用于台架定位排查。

**比赛安装已确认（2026-11-09）**：相机装在机械臂末端。仿真与实机入口均支持腕部模式，实机尚待标定及验证。末端版本需要拟合固定外参 `T_tool_from_depth`，用深度帧对应的关节反馈与 M3 正运动学计算 `T_base_from_tool`，再组合得到每帧的 `T_base_from_depth = T_base_from_tool @ T_tool_from_depth`。若手眼标定使用 RGB 图，应将 SDK 提供的深度到 RGB 外参串入该链。以静止多姿态拍摄和固定目标的跨姿态底座坐标一致性作为首轮验收。末端相机的视场、遮挡和支架碰撞需要在仿真与实机分别校验。

- **rectangle**：按桌面高度分割物块，拟合最小面积矩形，沿短边闭合。规则物块可先用这条路线排查定位误差。
- **ggcnn**：输入 300×300 深度图，输出抓取质量、夹爪闭合方向和开口。用桌面高度掩膜过滤中心，再检查深度、开口、工作区和 Roll。比赛物块的开口校验取分割区域在闭合方向上的投影跨度，原始网络宽度单独记录。
- **YOLO**：在已对齐到深度坐标的彩色图上检测 `artificial_object`，确认目标深度区域后再交给 GG-CNN。

## MuJoCo 仿真

Windows Python 3.12 x64 已验证。Ubuntu 服务器可以用相同命令运行无窗口批测，当前尚未在该服务器验证。安装完成后无需相机、串口、ROS 2 或 GPU。

```text
python -m venv .venv
```

Windows 用 `.venv\Scripts\Activate.ps1` 激活，Ubuntu 用 `source .venv/bin/activate`。然后在工程目录运行：

```text
python -m pip install -r requirements-sim.txt -r requirements-vision.txt
git clone https://github.com/waveshareteam/roarm_ws.git .vendor/roarm_ws
git -C .vendor/roarm_ws checkout 40dbd84b553695212fab713e8465f817ba95454d
python simulate.py --camera overhead --trace
python replay.py sim_output/report_0.npz
```

已有仓库时直接执行 checkout。回放需要桌面环境，批测只生成报告、深度预览和可选轨迹。仿真从官方 Xacro 读取关节、限位、网格和 `hand_tcp`，在 MuJoCo 运行数值 IK、接近位开始的下降和抬升轨迹、碰撞和接触动力学。俯视深度来自桌面和物块的射线投射，假设采集时机械臂停在视野外。相机噪声、遮挡、支架、实机惯量、摩擦和驱动响应尚未标定。平行夹指模式是用于验证感知和运动链路的物理简化模型，摆位地面真值只用于评分。

腕部回放可同时显示动态深度和保存的推理帧。动态窗口由当前姿态重新射线投射，保存窗口保留仿真当时的深度噪声和抓取矩形。

```powershell
python replay.py sim_output/ggcnn-clutter-wrist_0.npz --depth-view --inference-view
```

导入腕部标定时，给 `simulate.py` 加 `--wrist-config config.json --wrist-recording field-frame.npz`。仿真读取配置中的手眼外参和 ROI，从录帧取同一区域的深度射线，重新渲染仿真深度。轨迹保存这两套参数供回放使用。机械臂和相机外壳的碰撞几何仍是现有模型，MuJoCo 彩色视角也仍使用名义视场。该入口用于抓取对照，暂不与 `--verify-storage` 一起使用。

再加 `--wrist-workspace measured-workspace.json` 可使用扫掠审计里的 `wrist_envelopes` 和 `padding_mm` 替换名义相机与支架的碰撞盒。盒子按 link5 局部坐标放置，回放从轨迹恢复这些尺寸。碰撞检测使用加厚盒，深度射线只检查未加厚的外壳。若每个盒子都填写 `mass_g`，仿真按未加厚的均匀长方体估算腕部载荷重心和惯量。全部省略时保留名义质量，部分填写会报错。

噪声批测用 `--sensor-profile sensor.stress.json --sensor-seed 17 --repeats 5`。同一套场景和起始种子可以对照不同策略，报告保留每次试验的场景编号、重复编号和种子。含失败的批次仍写出完整报告，再以状态码 1 退出。

实测样本可用 `--case-manifest cases.json` 批测。清单是 JSON 数组，例如 `[{"id":"sample-a","xy_mm":[272,25],"yaw_deg":20,"size_mm":[40,28,30],"mass_g":20,"support_height_mm":55,"clutter":true}]`。XY 是底座系位置，支承面高度以地面为零。默认使用长方体。异形件可填 `"mesh_stl":"sample-a.stl"`，路径相对清单。STL 须为二进制、毫米单位、包围盒中心在原点，且外形尺寸与 `size_mm` 相符。使用闭合凸网格，MuJoCo 接触按凸包计算。回放需要原 STL 路径可访问。岩壁和搁台仍使用仿真场景的几何。

矩形搁台可另填 `"ledge_box_mm": [294, 5, 140, 180, 22]`，依次为底座系中心 X、Y、沿 X 的长度、沿 Y 的宽度和厚度。顶面由 `support_height_mm` 定义。清单会拒绝超出搁台边界的样本包络；岩壁本体仍使用默认几何。

| 实测配置 | 三种摆位结果 |
| --- | --- |
| 矩形基线，平行夹指 | 3/3，抬升约 66.6–66.7 mm |
| Cornell GG-CNN，平行夹指 | 3/3，抬升约 64.8–66.4 mm |
| 矩形基线，官方原始网格和转动夹指 | 0/3，闭爪后滑脱，未成功抬升 |

```text
python simulate.py --gripper vendor-mesh --report sim_output/vendor-mesh.json
python simulate.py --upstream .vendor/ggcnn --trusted-checkpoint .vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell --report sim_output/ggcnn.json
```

第二条命令需要先下载作者源码并解压官方权重。报告的 `passed` 要求物块通过短试提、全程抬升超过 50 mm 且保持双侧接触。任一摆位未通过时，命令写出报告后以状态码 1 退出。官方网格的碰撞凸包、单转动指闭合与实际夹指接触面都要进一步核对。原装夹爪仿真使用 `--gripper vendor-split`。

## 安装与离线演示

在本目录打开 PowerShell。建议用支持相机 SDK wheel 的 Python 3.11 x64。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe run.py --config demo_output/config.json --frame demo_output/frame.npz --preview demo_output/preview.png
```

输出包含抓取像素、底座系表面点、估计开口和四条计划指令。预览中的绿框是抓取矩形，白线是夹爪闭合方向。运行双模型时安装 `requirements-vision.txt`。接入实机时，再按具体型号安装对应相机 SDK。

## 相机与外参标定

1. RealSense 安装 `pyrealsense2`。Orbbec 按具体型号和固件选择官方 SDK v2 或 v1，确认对应 SDK 的深度示例能取到 Y16 数据。本项目适配 v2 的 Python API（发行包 `pyorbbecsdk2`，导入名 `pyorbbecsdk`），对 Brown-Conrady/rational 深度镜头使用 SDK 提供的内参和畸变系数。建议先接一台相机，避免误选设备。
2. 固定相机和机械臂底座。运行 `python capture.py realsense point1.npz`，或将参数替换为 `orbbec`。点击保存帧上的标定点，终端打印像素与光学坐标。采集期间保持工作台及相机固定。
3. 在底座系中量出同一物理点的 XYZ（mm）。可用已测量的标定治具，或在低速手动点动下用已知 TCP 的探针触碰该点并读取 `T=105`，把探针尖端偏移计入坐标。深度采样时把探针移开，点应位于平坦表面内部，远离边缘。建议采 6–10 个分布在工作区不同位置与高度的拟合点，再留至少 2 个独立验证点。
4. 建立 `pairs.csv` 和 `check.csv`，两者表头均为 `frame,u,v,x_mm,y_mm,z_mm`。frame 为相对 CSV 的 NPZ 路径。运行 `python calibrate.py pairs.csv camera_to_base.npy --check check.csv --max-error-mm 5`。查看每点残差并按夹取公差调整阈值。这里拟合的是 `p_base_mm = R @ (1000 * p_camera_m) + t_mm`。换相机、挪动支架或改变深度采集设置后重新校验。
5. 将 `config.example.json` 复制为 `config.json`，填入外参路径、方形 ROI、桌面在底座系中的 Z 和物块高度范围。执行 `python run.py --camera realsense --preview preview.png`。把输出的 XYZ 和已知物块位置对照，测试工作区中心与边缘，连续多次测量同一块的坐标散布。

## 用 Cornell 权重运行 GG-CNN

从作者的 [GG-CNN 仓库](https://github.com/dougsm/ggcnn) 获取源码，从 [Cornell 权重发布包](https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn_weights_cornell.zip) 解压选择原版 GG-CNN 模型文件。使用该虚拟环境安装 PyTorch。官方发布模型使用 Python pickle 对象，只向 `--trusted-checkpoint` 提供可信的原始文件。此入口读取作者发布的完整模型对象，训练源码、权重与 Cornell 数据集由上游提供。

```powershell
python run.py --camera realsense --backend ggcnn --upstream ..\ggcnn --trusted-checkpoint .\weights\model.pt --preview ggcnn.png
```

将示例权重路径改成解压后的实际文件。也可以给相同的 `--frame` 分别运行两个后端，比较同一帧输出。GG-CNN 的质量分数用于排序和阈值筛选。实机流程会在悬停位重新取帧，检查目标稳定后再下降，机械臂执行时读取位姿反馈。RSS 论文中的视觉闭环还会在接近目标时反复更新抓取候选。

## 实机联调顺序

### 未标定时的实时双模型预览

本机简短启动方式（默认 Gemini 335L、现有 YOLO 和 Cornell 权重）：

```powershell
cd D:\codex\lab\Star\roarm_grasp
python .\start_preview.py
```

也可运行 `python debug.py` 进入命令菜单：`v` 预览、`p` 列出串口、`c` 保存单次 RGB-D、`q` 退出。`python debug.py v` 可直接打开预览。实时预览与单次采集不要同时占用同一台相机。

仿真短命令：`python debug.py s` 跑原装分段夹爪与 GG-CNN，`python debug.py r` 回放最近一次 s 的轨迹；`python debug.py sy` 再加入真实 YOLO 权重做语义筛选。每次仿真新建独立输出目录，sy 不替换 r 使用的 s 轨迹指针。2026-11-11 的实跑中，s 的三组 baseline 为 3/3；同模型与摆位开启 YOLO 后为 0/3，均因无检测框返回 `semantic_target_not_confirmed`，没有开始抓取。仿真 RGB 为简化材质，模型权重对实际比赛目标的适用性仍需真实样本验证，不能把 s 的结果视为双模型或 Sim2Real 验收。

语义仿真现默认 `--rgb-renderer mujoco`，渲染光照、阴影和可见网格，再按目标光学射线重采样到深度光栅；深度仍由现有碰撞几何射线求交生成。两者的可见网格/碰撞代理差异、真实目标纹理、镜头响应与相机噪声仍待实测。可用 `--rgb-renderer flat` 重现之前的几何颜色诊断。投影已用已知偏轴方块核对；接入光照渲染后同一批三工况仍为 0/3（YOLO 无检测框），不能据“渲染更好看”推断识别或真实抓取通过。

按 S 保存的数据可用 `python start_preview.py --frame field_recordings/<时间戳>/rgbd.npz` 离线重放推理。预测 JSON 记录检测/抓取权重的 SHA-256、类别映射和相机后端。检测框与抓取像素属于 `aligned_rgb.png` 和深度光栅，不能直接作为 `original_rgb.png` 的标注；原始彩色图尺寸和投影通常不同。

Python 需使用本机已部署的 Anaconda Python 3.12；启动脚本自动加入工程本地依赖目录。额外参数可追加，例如 `--device cpu`。原始 RGB 会单独显示，主窗口使用对齐至深度光栅的 RGB，避免混用两个坐标系绘制检测框。按 S 保存当前观测到 `field_recordings/<时间戳>/`，包含原始 RGB（可用时）、对齐 RGB、RGB-D NPZ、预览和预测 JSON；这些观测未人工审核，不自动生成训练标签。没有检测到目标不等于这张图已被证明是负样本。

当前接入相机为 Gemini 335L（2026-11-11 实测 RGB-D 为 848×480）。在本机 PowerShell 运行：

```powershell
cd D:\codex\lab\Star\roarm_grasp
$env:PYTHONPATH="$PWD\.deps;$PWD\.orbbec-python-deps;$PWD\.torchdeps;$PWD\.semdeps"
& D:\anaconda\python.exe .\preview_live.py --camera orbbec --target-weights "D:\Lab\挑战杯\LunarRover_HUST-master\weights\best.pt" --upstream .vendor\ggcnn --trusted-checkpoint .vendor\cornell\ggcnn_weights_cornell\ggcnn_epoch_23_cornell
```

左侧为已对齐至深度光栅的彩色图，右侧为深度图（0–2 m 映射到颜色，黑色为无效/超出预览有效范围的深度）。黄框为 YOLO 目标，蓝框为逐目标 GG-CNN 输入区域，绿矩形为网络图像抓取候选。没有目标、深度空洞过多或无有效候选时不画抓取矩形。Q/Esc 退出。可用 `--frame saved.npz --headless --output preview.png` 离线排查。

这个入口不打开机械臂串口，不生成运动命令，也不需要手眼外参。YOLO 框内部只是候选搜索区域，尚未做支承面分割、夹爪开口和碰撞验收，不能据此认为候选可执行。真正抓取仍使用下述 `run.py` 的标定、目标分割和部署验收路径。更换夹爪后还需同步模型、TCP 和开合标定。

正式抓取使用 YOLO 确认特殊人造物，再用 GG-CNN 计算夹取姿态。在工程目录运行下列命令定义模型参数，把路径改成部署机器上的实际位置。

```powershell
$detector = "D:/Lab/挑战杯/LunarRover_HUST-master/weights/best.pt"
$ggcnnSource = ".vendor/ggcnn"
$ggcnnWeights = ".vendor/cornell/ggcnn_weights_cornell/ggcnn_epoch_23_cornell"
$pickModels = @("--target-weights", $detector, "--upstream", $ggcnnSource, "--trusted-checkpoint", $ggcnnWeights)
```

- 用官方界面验证当前固件支持 `T=104`、`T=105` 和 `T=106`。关闭占用串口的其他程序。可用 `python -c "from arm import Arm; a=Arm('COM5'); print(a.feedback()); a.port.close()"` 读取状态。
- 在桌面上方测试垂直向下的 Pitch、Roll 零位、方向符号和机械限位。Roll 使用 `r = sign * (夹爪闭合轴在底座 XY 平面的角度 - atan2(y,x)) + zero`，由实测确认适用于当前工具安装。用两种物块朝向、两个底座方位验证。实测夹爪开合角和实际最大开口，填入配置。
- `surface_to_tcp_mm` 是从可见顶面 Z 到固件 TCP 抓取 Z 的带符号偏移。结合夹指长度、夹取深度和 TCP 定义实测。用首轮正偏移或提高测试工位，让夹爪在物块上方空跑并检查方向与位置。验证工作区和起始姿态的实际通路，包括手腕和夹指的扫掠空间。
- 配置内的尺寸和角度均为示例。确认工作区、起始姿态与路径后设置串口 `port` 和 `commissioned: true`。清空机械臂扫掠范围，保持可直接切断电源，从泡沫块低速开始。先执行 `python run.py --camera realsense --hover-only`。用空夹和人工确认夹住比赛物块时的反馈标定 `held_angle_rad` 和 `held_min_load`，区分空夹、持物与异常卡住。填入实测范围后才能运行 `python run.py --camera realsense --execute @pickModels`。
- 程序执行接近、悬停复测、下降、闭爪、抬升。悬停时若目标偏移超过 8 mm、Roll 改变超过 10° 或连续三帧表面点漂移超过 3 mm，则在下降前终止。闭爪和抬升后分别检查夹爪角度及负载，负载只是持物的间接证据。移动阶段检查位置与姿态反馈，超时即停止发送后续命令。工作区校验针对 TCP 目标，机械臂固件负责 IK，路径与整臂碰撞须在台架验证。动作完成后人工确认物块是否持住。

## 文件分工

| 文件 | 职责 |
| --- | --- |
| camera.py / capture.py | 两类相机的深度采集、射线映射和点选录帧 |
| geometry.py / calibrate.py | 深度反投影、刚体外参拟合与独立验证 |
| perception.py / ggcnn.py | 矩形基线与可替换的学习抓取后端 |
| planning.py | 像素方向转底座系，开口、Roll 与目标点检查 |
| arm.py | USB 串口 JSON 协议与反馈 |
| run.py / demo.py | 实机工作流入口与离线合成场景 |
| sim_model.py / sim_camera.py | 官方模型转换、物理场景及模拟深度 |
| sim_motion.py / simulate.py / replay.py | 关节限位 IK、接触验收、无窗口批测和图形回放 |

参考资料：[RoArm-M3 产品 Wiki](https://www.waveshare.net/wiki/RoArm-M3) · [微雪 JSON 指令](https://docs.waveshare.net/RoArm-M3/JSON-Command-Control/) · [GG-CNN 论文](https://arxiv.org/abs/1804.05172) · [Orbbec Python SDK](https://github.com/orbbec/pyorbbecsdk) · [RealSense SDK](https://github.com/realsenseai/librealsense)


仿真参考：[微雪 ROS 2 官方模型](https://github.com/waveshareteam/roarm_ws/tree/40dbd84b553695212fab713e8465f817ba95454d) · [MuJoCo 建模文档](https://mujoco.readthedocs.io/en/stable/modeling.html)

## 比赛版腕部相机仿真

`simulate.py` 默认使用 `--camera wrist`。旧的固定俯视基线可用 `--camera overhead` 重跑。腕部模型把橙色相机壳、灰色支架、光学窗口和名为 `wrist_depth` 的传感器绑定到官方 M3 的 `link5`。壳与支架参与碰撞。当前相机尺寸、质量、65 mm 侧向安装偏移和 58° 视场是待替换的名义参数，集中在 `sim_model.py`。深度射线包含机械臂和场景几何，因此夹爪进入视野会产生遮挡。传感器暂未建模有效距离、空洞和噪声。

```powershell
$env:PYTHONPATH = "$PWD\.simdeps;$PWD\.deps"
python simulate.py --camera wrist --trace --report sim_output/wrist.json
python replay.py sim_output/wrist_0.npz
python replay.py sim_output/wrist_0.npz --wrist-view
```

`wrist_depth_0.png` 是实际参与预测的深度图，绿框表示抓取方向和估计开口，黄十字表示主点。对应 NPZ 含米制原始深度、像素射线和拍照时相机到底座的外参。第三人称回放显示实体相机，`--wrist-view` 切换到相机光学视角。

三个摆位从同一预设观测位开始。规划使用拍照时模拟相机姿态和深度，物块真值只用于建场景与评分。简化平行夹指的矩形和 Cornell GG-CNN 路径均完成 3/3 抬升。用 `--gripper vendor-mesh` 测原装转动夹爪，腕部相机版本仍是 0/3，因而原装夹爪抓取尚未通过迁移验收。实际比赛版本还需以具体相机和支架参数替换名义模型，并对照实测手眼外参、相机深度有效范围、固件关节映射、TCP、夹爪接触和路径间隙。

比赛任务与仿真验收依据见 [XH-202605.md](XH-202605.md)。


## M3-S 腕部相机实机标定

### AprilTag 原始观测采集

`python debug.py board` 检查 tag36h11 ID 与7×5棋盘内角点；它不连接机械臂。当前用户选择50毫米 Tag ID 0。`python debug.py tag` 打开 Gemini 335L 和 COM6，按 F 切换拟合组、C 切换独立检查组、S 保存静止观测、Q 退出。每次运行新建 `calibration/tag-<时间戳>/`，保存原始 RGB、工厂 RGB 内参和畸变模型、深度到 RGB 的外参（毫米）、深度射线、五关节角及原始 RGB Tag 角点。脚本不发送运动命令，但打开串口可能复位控制板并触发固件启动动作，因此操作前清空运动范围。

底座和Tag板在整个采集期间保持不动；通过网页低速点动改变腕姿态并停稳后保存。建议至少10组拟合、4组独立检查，包含不同朝向和位置，避免只平移或只绕同一轴旋转。此入口当前只采集，尚未证明相机外参、关节映射或抓取可用。RGB手眼结果后续须串入保存的深度到RGB外参才能用于深度定位，不能把RGB相机位姿直接当作深度相机位姿。

静止采集会在五次取帧之间逐次读取五关节反馈，检查整个采集窗口内的角度范围（默认 0.003 rad），再用最后一帧前后的反馈估计该帧姿态；跨越 ±π 时按连续角度处理。此检查能拒绝采集中途移动后返回原位的情况，但不证明传感器/固件的实际时间同步，也不证明拆除关节后正运动学仍有效。当前缺少最终安装状态的实测外参、独立验证和完整夹爪配置，禁止把示例配置直接设为已部署。

`capture.py` 支持 `--points-csv` 和 `--point-base-mm X Y Z`：使用已独立测量的底座系靶点坐标，点击深度窗口或彩深叠加窗口上的同一个物理点后，自动向 CSV 写入图像相对路径、像素和实测 XYZ。每帧只保存一个点，重复使用帧和覆盖录帧会被拒绝。拟合组与独立验证组使用不同录帧和 CSV。该功能只记录数据，不移动机械臂；已知点坐标不能用未验证的机械臂自身反馈代替。外参采集前需确认相机固定在哪个连杆，当前求解默认固定在 `link5`，并确认五关节反馈仍完整可用。

1. 复制示例配置，设置 camera_mount 为 wrist，填写串口和官方模型路径。joint_signs 和 joint_offsets_rad 是固件关节反馈到官方 Xacro 关节的映射。示例为单位映射，须在实机逐关节核对零位、方向和基座原点后使用。
2. 手动点动到多组分散姿态，停稳后用 python capture.py orbbec point1.npz --arm-port COM5 录制。也可用 realsense。文件会保存深度、射线和拍照前后检查过的五关节角。用已测量底座系坐标的标定点填写原有 CSV 格式。训练点与验证点应包含不同姿态，且验证姿态独立采集。
3. 运行 python calibrate.py pairs.csv link5_from_depth.npy --check check.csv --wrist-config config.json --max-error-mm 5。将配置 transform 指向这个文件。5 mm 是初次调试阈值，最终按夹具间隙和误差预算确定。
4. 保持 commissioned 为 false，先运行 python run.py --camera orbbec --preview preview.png。此模式只读机械臂反馈并输出计划。人工换位后，核对同一标定点的底座系坐标。重复验证之后再进入已有悬停与抓取联调流程。

当前对照验证了 25 组相同关节输入下，独立 FK 与 MuJoCo 的腕部相机位姿一致。这个对照验证了模型数学，固件映射、实机外参与夹爪接触仍待台架数据。静止采集只检查帧前后两次反馈，不用于机械臂运动中的同步。

## 原装转动夹爪接触仿真

使用 vendor-split 将微雪官方固定指所在网格和转动指网格沿指身分段生成凸碰撞体。保留官方转动关节、外观网格和腕部相机。整块凸包会跨过指身凹处，导致物块从一侧被推出。报告记录闭爪后的接触点、法向及物块平移，并按实际夹指归并碰撞片。

```powershell
python simulate.py --gripper vendor-split --suite envelope --trace --report sim_output/wrist-vendor-envelope.json
python replay.py sim_output/wrist-vendor-envelope_0.npz
```

Windows 仿真结果为原三组轻木块摆位 3/3，矩形与 Cornell 权重分别验证。赛规尺寸和质量边界新增的三个压力场景尚未通过。50 mm / 300 g 方块及 45 × 30 × 38 mm / 300 g 物块未有效持住，20 mm 高的短物块触发固定指撞台。压力场景按赛规包络设置，实际材质、夹指表面、驱动输出及舵机动态尚未实测。整块凸包的 vendor-mesh 与简化平行夹指的 parallel-pads 均可用作对照。

## 桌面夹指扫掠检查

sim_planning.py 在仿真执行前根据支承面上方剩余空间限制下探量，再用独立 MuJoCo 状态采样检查张爪转场、竖直下降和原装夹指闭合。无法执行的候选会被跳过。原装夹爪左右不对称，因此检查相反朝向和固定指侧向余量。此处是离散碰撞检查，物理步进仍检查执行中的碰撞。真实路径和支承面外参需要实机验收。

同一尺寸与质量工况复测见 sim_output/wrist-clearance.json。短木块下探从原 16 mm 减到约 9.8 mm，成功抬升。原三组轻木块仍通过，两组 300 g 都形成双侧接触但随后丢失，记录为 lost_after_close。仿真参数未用于推断真机夹持力。

## 转动夹爪扭矩上限与抬升诊断

原转动夹爪仿真曾沿用整臂占位上限 18 N·m。当前改为微雪 M3-S EoAT 标称 20 kgf·cm，对应约 1.96 N·m，作为仿真上界。可用 --jaw-limit-nm 设更低的值模拟可用扭矩下降。铭牌值与仿真接触力不能当作实测持续夹持力。其他关节的仿真驱动限值仍待标定。微雪还给出 0.5 m 处有效载荷 0.2 kg。赛规样本上限约 0.3 kg，实际带腕部相机的作业半径与夹持能力需分开测定。

压力测试报告中的 grip_history 含每个抬升路点的夹爪位置、关节驱动力、两指法向力和切向力，以及物块与 TCP 高度。first_non_bilateral_sample 是首次采样到双侧接触丢失的位置。rated-lift-12.json 和 rated-lift-48.json 用相同场景比较抬升路点停留时间，两组均为 4/6。两种 300 g 工况都丢失双侧接触。延长每个路点停留时间未恢复持物。这里的对照仅支持排除这次单一减速方案，不能由仿真反推出实机失效根因。

来源：[微雪 M3 系列手册](https://www.waveshare.com/wiki/RoArm-M3) · [微雪 M3-S EoAT 规格](https://www.waveshare.com/roarm-m3-s-ai-kit.htm)

## 采样入仓仿真

--storage 会在腕部视角仿真中加入一个内径 110 × 110 mm 的固定料仓。位置为底座系 (220, -140) mm，属于工程样机参数，待替换为实际车载料仓 CAD 和安装位姿。该模式执行抬升、带物转运、释放和停稳验证。转运先用夹持相对位姿检查物块和夹爪扫掠，物理执行仍依赖接触和摩擦。腕部朝向允许沿转运路径渐变以绕开腕部关节限位。释放前检查张爪扫掠，释放后要求物块完整落在仓内并接触仓底且停稳。

```powershell
python simulate.py --gripper vendor-split --storage --suite envelope --trace --report sim_output/storage-envelope.json
python replay.py sim_output/storage-envelope_0.npz
```

矩形基线四个轻样本完成入仓，两件 300 g 样本在入仓前滑脱。Cornell GG-CNN 原三个轻样本完成入仓。该链路还需要实际料仓几何、实机入仓确认及整车运输和指定区域卸载验收。

## 岩壁台面工况

--suite ledge 在同一模型里加入岩壁和水平支承台。规划从已知场景几何读取台面高度，腕部深度负责估计样本位置。台面与岩壁参与碰撞，原装夹爪取样后可以用 --storage 验证入仓。

```powershell
python simulate.py --gripper vendor-split --suite ledge --storage --trace --report sim_output/ledge.json
python replay.py sim_output/ledge_0.npz
```

35 mm 和 55 mm 高台面的矩形后端均完成入仓，Cornell GG-CNN 在其中一个姿态闭爪时未形成双侧接触。200 mm 高台面与当前预设观测位相交，在推进物理前拒绝，报告 observation_pose_blocked 和可由上层处理的换观测位或重定位提示。该报告退出码为 1，因为场景集中包含未完成抓取的工况。

台面高度相对场地地面计，台面位置和岩壁均为测试几何。此场景覆盖从水平岩壁搁台取样。没有实测岩壁地图，也没有实现在线估计支承面、自动底盘重定位或直接从倾斜支承面取样。

## 从腕部深度估计水平支承面

仿真和实机入口共享 support_plane.py。先将腕部深度点变换到底座系，再从搜索区域估计近水平支承高度。高度竞争、覆盖面积不足或深度点不足会拒绝计划。仿真报告记录估计高度、点数及与仿真几何的误差；几何真值只用于报告，不用于计算下探。

实机可在配置中填写 support_search_mm（底座坐标的 XYZ 下界、上界，单位 mm）与实测 tcp_support_clearance_mm。搜索区域应覆盖一块可见的地面或水平搁台，并排除其他高度的支承和车体。默认仍用已知 table_z_mm。实机开启估计后，规划在初拍及悬停复测时分别估计，并限制下探，不满足剩余夹取深度时拒绝。该功能覆盖单一近水平支承，倾斜支承和未知场景的全局表面提取尚未接入。

## 碰撞筛选与腕部观测位恢复

腕部仿真从模型的初始关节位开始，按 sim_observation.py 的姿态库筛选观测位。对候选位逐一做逆解，检查从初始位到候选位的张爪路径，并确认腕部视野中存在可估计的支承面及前景。选定后才在物理仿真中执行观测路径。姿态库与仿真样本位置独立。观测姿态允许沿机械臂前向倾斜，实际抓取仍按竖直下探规划。

ledge-views.json 中，200 mm 台面的原俯视路径被碰撞检查拒绝，随后斜视位成功取得深度并定位。当前竖直接近姿态的逆解失败，返回 grasp_path_rejected 和 change_approach_or_reposition_base。低台面 2/2 与地面 3/3 入仓复测通过。高台面的轨迹和深度可分别查看 sim_output/ledge-views_2.npz 与 sim_output/ledge-views_depth_2.png。上层提示尚未连接到实际底盘，实机也尚未配置或验证这套观测动作。

## 高搁台斜向接近

岩壁工况保留竖直接近作为首选。逆解或扫掠不通过时，尝试绕机械臂径向向前倾斜，沿夹爪轴线接近。候选仍检查关节限位、张爪转场、接近及闭爪扫掠。200 mm 搁台在矩形基线下找到 0.65 rad 的斜向路径并执行到闭爪。闭爪未形成双侧接触，因此停止在抬升前。低搁台和地面工况沿用竖直路径并完成入仓。可用 replay.py sim_output/ledge-tilt_2.npz 查看高台失败轨迹。当前接触门控和斜向轨迹在仿真中运行，未接入实机。

## 深度与手眼误差压力测试

sensor.stress.json 是可复现的工程扰动配置，未绑定到具体相机型号。可以修改有效距离、深度偏差、噪声、随机失效、成片空洞和手眼扰动。--sensor-seed 固定随机序列，报告保存每个工况的实际种子与配置。观测位预筛用理想光线检查几何可见性，选定观测位后送入抓取链的是受扰动的深度与估计外参。物块、机械臂和碰撞几何不受外参扰动影响。

```powershell
python simulate.py --gripper vendor-split --storage --sensor-profile sensor.stress.json --sensor-seed 17 --report sim_output/stress-17.json
```

两个种子的矩形后端分别通过三个地面轻样本。种子 17 的 Cornell GG-CNN 也通过三个工况。最小有效距离设为 450 mm 时，这套近距离观测在抓取规划前因深度不足被拒绝。后续需从真实腕部深度帧估计噪声、距离与空洞分布后替换配置。

## 实测深度配置

用固定相机和独立测量过的平面靶采集近、远和中间距离。平面在相机光学坐标系中满足 n · p = d，法向为单位向量，d 用毫米。正对光轴的平面可用 [0, 0, 1, d]。ROI 只圈靶面内部，排除边缘和背景。相机光学原点到靶面的距离要单独测量。

```powershell
python capture.py orbbec near.npz --depth-stack 30
python capture.py orbbec far.npz --depth-stack 30
python capture.py orbbec check.npz --depth-stack 30
python characterize_depth.py target-captures.json sensor.measured.json --report depth-characterization.json
```

target-captures.json 的 fit 数组至少有两项，check 至少一项。每项含 file（相对清单路径）、plane（法向和距离）和 roi（x、y、宽、高）。验证集应独立重新摆放采集，距离落在拟合覆盖范围内。程序比较绝对误差、偏差、噪声和失效率，通过后写出可交给 --sensor-profile 的配置。失败时仍写报告，已有配置不会被覆盖，部署前要核对本次报告。导出的距离范围仅是采集覆盖区间。随机失效是像素级近似，成片空洞和手眼误差需另测。

## 腕部相机实机定位验收

完成手眼拟合后，用独立录帧进行实机定位验收。CSV 表头为 target_id,frame,u,v,x_mm,y_mm,z_mm。至少选择两个不同的已知底座坐标靶点，每点至少用三个不同静止腕姿态采集。可用同一帧标两个靶点。所有录帧须来自部署使用的同一深度流。用 capture.py 的 --arm-port 录制关节反馈，且不要复用拟合手眼外参时的录帧。

```powershell
python validate_wrist.py config.json acceptance-points.csv wrist-acceptance.json --camera orbbec
```

默认逐点要求最大绝对误差 5 mm、同一点跨姿态最大两两差 4 mm。还需根据夹指余量设更严门限。报告通过后，把 wrist_acceptance 指向该报告。腕部模式下，只读预览可继续使用，--hover-only 和 --execute 则须通过报告绑定检查。修改外参、官方 Xacro、关节映射或更换深度射线后需要重测。报告只验收静止腕部定位，不证明路径安全或实际持物。换相机支架、碰撞或重装后应主动重新标定和验收，即使文件指纹没有变化。

## 真机短试提与反馈日志

--execute 在闭爪反馈合格后先执行 proof_lift_mm 的短试提，默认 12 mm，速度命令最高为 0.10。试提高度必须大于两倍位置容差，并低于全程抬升高度。机械臂需通过试提位置反馈和三次持物反馈，才继续全程抬升。试提失败时停止发送后续动作，不自动重抓或放爪。夹爪角度与 tG 是间接持物证据，需在实际样本上校准并对照视频或独立持物传感器。

```powershell
python run.py --camera orbbec --execute --motion-log bench-pick-001.jsonl @pickModels
```

首次带物台架验证可在同一命令中加 `--proof-lift-only`。通过短试提后机械臂留在试提位并保持夹持，记录 `proof_lift_telemetry_passed`。该模式适合在台架旁观察物块是否真的离开支承面，以及是否出现滑移。

再加 `--record-observations field-run-001` 会保存 `003.npz`（第三次悬停复测）和 `proof-lift.npz`（试提后）。如果两帧都能看到同一个物块特征和静止参照点，可在深度图上分别标出它们的像素，然后用 `check_proof_lift.py` 计算底座系位移。每组像素按试提前的 u、v 和试提后的 u、v 填写。下列像素仅为命令格式示例。

```powershell
python check_proof_lift.py config.json field-run-001/003.npz field-run-001/proof-lift.npz --object-pixels 130 71 128 75 --reference-pixels 92 110 90 114
```

日志使用新文件，记录抓取段的命令、到位反馈、持物采样以及中止原因。悬停复测仍在抓取段开始前。试提和全程抬升的持物检查均在动作到位后执行。

## 夹爪反馈实测验收

可以用 grip_calibration.py capture COM5 held trial.json 只读记录夹爪反馈。标签为 held、empty、jam，由独立视频或人工观察确认。采集模块只发送 T=105 查询，不控制开合。空夹、持物和已有安全异常记录应覆盖实际夹爪、样本和指令设置。

采集清单 JSON 含 fit 和 check 数组，每项为 {"file": "trial.json"}。每组都要有 held、empty、jam，至少两次独立持物试验。运行 python grip_calibration.py fit config.json grip-captures.json grip-acceptance.json。闭合角和负载门限从拟合组持物数据得到，两组的持物样本均需命中，空夹或卡住只要有一条样本命中就判失败。通过后把配置 grip_acceptance 指向该报告，自动抓取即使用报告门限，并校验开合角仍与验收时一致。旧配置仍可使用手填门限。反馈验收只证明所测条件下可区分状态，未采集的形状、夹持姿态和异常仍需验证。

## 实机示教转运与释放

默认执行到抬升。腕部模式可加 --store 走单独调试过的料仓路线。storage 在示例中为 null，实际配置需填 commissioned、lift_entry_workspace_mm、bin_xy_mm、rim_z_mm、tcp_rim_clearance_mm、release_margin_mm、waypoints。前两个区域用坐标下界和上界表达，仓口只用 XY。waypoints 逐项为 [x_mm,y_mm,z_mm,pitch_rad,roll_rad]，最后一点在仓口内侧。最小仓沿间隙与仓口内缩均需大于到位容差。要在空车上验证抓取抬升区域到第一路点，以及所有相邻路点的整臂路径、相机与料仓间隙，再设置 storage.commissioned。

```powershell
python run.py --camera orbbec --execute --store --motion-log bench-store-001.jsonl @pickModels
```

计划会在抓取前检查入仓路线。转运路点逐个检查持物反馈，只有实际释放位仍处于仓口允许范围才允许开爪。成功输出 released_unverified，表示开爪反馈确认，需独立观测确认物块落在仓内。TCP 路点校验不覆盖整个机械臂扫掠，也不等价于整车卸载。实机路线未在当前工作区执行。

## 腕部深度入仓复核

在实机 storage 配置中设置 verify_depth 为 true，并提供实测 floor_z_mm。可选 verification_waypoint 与转运点格式一致，用于释放前持物拍一帧、回到释放位开爪，再从相同观测位拍三帧。使用独立观测位时，必须预先验证释放位与观测位之间带物和开爪两种状态的整臂间隙。仿真中原先的释放视角不足以可靠看到仓底，宜先用只读预览确认视野。

bin_depth.py 将每帧转换到底座系，对比仓内 5 mm 网格的可见高度。只有新出现的内部连通区域在连续三帧稳定，且视野和仓沿条件满足，才输出 deposit_depth_confirmed。遮挡、移动、空仓和贴边变化保留 released_unverified。它提供可见新物的证据，无法识别物品类别或观测被遮住的部分，不能替代整车卸载验证。当前已通过合成反例和 MuJoCo 仓内深度帧验证，未运行实机。

## 入仓深度复核的完整物理循环

使用 --storage --verify-storage，仿真在抓取后先带物到独立腕部观测位拍参考帧，再到仓口开爪，最后张爪回到同一观测位拍三帧。带物路径预检使用当前物块相对位姿检查扫掠，物理执行使用原装夹爪接触；失去双侧接触就中止。复核的最终通过条件同时需要物理落仓与和实机入口共享的深度复核通过。无配置时使用理想深度。指定 --sensor-profile 后，抓取观测和仓内复核都应用扰动。同一轮固定相机外参误差，各帧独立生成深度误差。

```powershell
python simulate.py --gripper vendor-split --storage --verify-storage --trace --report sim_output/storage-depth-cycle.json
python replay.py sim_output/storage-depth-ggcnn_0.npz
```

矩形基线 2/3，另一个摆位在带物到料仓相关位置停稳时失去双侧接触，未执行释放。Cornell GG-CNN 3/3 同时通过物理落仓和深度验证。复核位和料仓位置是仿真几何，需要用实际仓体、实际传感器视野和实机间隙重测。


### 仓内深度扰动复测

```powershell
python simulate.py --gripper vendor-split --storage --verify-storage --sensor-profile sensor.stress.json --sensor-seed 17 --trace --report sim_output/storage-depth-stress-17.json
```

两个随机种子的矩形基线均为 2/3，落仓的四次均通过深度复核。另两次失去双侧接触并在释放前中止。两个同样扰动的空仓对照均未产生入仓证据。工程扰动参数尚未由实装相机标定。

## 目标类别确认

可复用团队 LunarRover 工程的本地 YOLO 检测权重。仅使用可信权重，在同一 Python 环境安装 ultralytics>=8.3,<9。RealSense 和奥比中光都将彩色图对齐到原始深度坐标，保留原深度射线和手眼外参定义。capture.py 的 --color 会同时展示彩深叠加窗口，实机可用靶板边缘检查配准偏差。

```powershell
python capture.py orbbec rgbd.npz --color
python run.py --camera orbbec --target-weights "D:/Lab/挑战杯/LunarRover_HUST-master/weights/best.pt" --preview semantic.png
```

自动抓取只接受特殊人造物（权重标签 artificial_object）。画面允许出现任意数量的物块，从通过深度定位和路径检查的特殊人造物候选中按抓取分数选一个，每次触发完成一件。检测框至少覆盖深度连通域的 75% 才允许送给矩形或 GG-CNN 后端。其他类别的检测框覆盖该连通域达 35% 时否决。悬停复测仍须找回同一件目标。预览与台架悬停可单独用于诊断其他类别。

RealSense 和奥比中光均接入彩色到原始深度坐标。奥比中光使用 SDK v2 的软件 C2D 和同步完整帧，缺失或尺寸不一致时拒绝该帧。两款相机的彩深对齐均尚未在设备上验证。离线 NPZ 可包含已配准的 uint8 BGR 图像，使用与该帧对应的相机到底座固定变换做复测。同一视觉权重在比赛照明、岩石和异形样本上的误检率需要实际采集验证。

单个检测框若同时覆盖两个或更多深度连通域各自至少 75% 的像素，该框无法为其中任何一个物体单独确认类别。同画面中有独立匹配的特殊人造物仍可进入规划。实拍验收应包含紧邻物块，检查深度连通域是否能分开。


## 腕部实机录帧的离线复查

```powershell
python capture.py orbbec scene.npz --arm-port COM5 --color
python run.py --config config.json --frame scene.npz --preview replay-preview.png
```

腕部配置使用录帧时保存的 joints_rad 和当前手眼外参恢复拍照时的相机位姿，重跑同一套支承面估计、类别确认和抓取规划。需要测试类别时，追加与实机相同的 --target-weights 和 --target-class。录帧缺少关节角时拒绝腕部回放。离线回放不会进入串口执行路径，不能配合 --execute 或 --hover-only。复测应保留未通过的帧，包括视野边缘和深度质量不足的观测。

腕部相机在线运行时可用 `--record-observations` 保存程序实际使用的观测，即使该帧没有通过目标筛选也会保留。目录必须是新的。`000.npz` 是初拍，后续编号按采集顺序递增。文件包含原始深度、射线、拍照对应的关节反馈，以及开启 YOLO 时的对齐彩色图。

也可以只采集、不启动推理。无桌面的 Ubuntu 服务器加 `--no-preview`，例如 `python capture.py orbbec field-frame.npz --arm-port /dev/ttyUSB0 --color --no-preview`。串口参数改为实际确认的设备路径。

```powershell
python run.py --camera orbbec --record-observations field-run-001 @pickModels
python run.py --frame field-run-001/000.npz --preview field-run-001/grasp.png @pickModels
```

## 指定目标抓取的录帧验收

evaluate_recordings.py 批量重跑与实机共用的离线规划。清单是 JSON 数组，每项为 {"frame":"relative.npz","condition":"low_light","targets":[[u0,v0,u1,v1]]}。框标在深度像素坐标，右下界不含端点。只标本轮允许且可抓的目标。岩石、空场景和只有其他类别的帧使用空数组。尽量使用紧框，并排除标注重叠导致无法区分的样本。数据须来自未用于训练或调阈值的独立录帧。

可为每个目标框独立标注一个或多个可接受的图像闭合轴，例如两个框对应 `"target_axes_deg": [[15, 105], [-20]]`。角度由图像向右朝向下为正，轴反向 180° 等价。若最佳候选与最近的标注方向相差超过 10°，记录为 `wrong_grasp_axis`。未标角度的帧仍只验收目标选择。

有独立测量的抓点时，还可以按目标框顺序填写 `"target_surface_mm": [[245, 35, 60]]`，单位为底座系毫米。报告输出 XYZ 偏差和总距离，默认允许 5 mm，也可通过 `surface_tolerance_mm` 单独设置。参考点是可见表面上的指定抓点，测量应独立于被验收的深度相机。

```powershell
python evaluate_recordings.py config.json field-manifest.json field-report.json --target-class artificial_object @pickModels
```

报告分别统计命中目标、错误目标、无目标却规划抓取、漏抓、正确拒绝和无效录帧。录帧缺失或格式错误不计为正确拒绝。严格通过要求所有有效标注帧结果正确且没有无效录帧。可追加 --upstream 和 --trusted-checkpoint 比较同批帧的 GG-CNN。结果证明抓点像素与标注框的一致性，实机接触、持物和入仓仍需单独测试。

悬停复测先按原抓点的底座坐标和已确认类别筛选候选。距离原点超过 8 mm 或类别变化的候选不能替换原目标。若多个不同的语义已确认深度连通域都能匹配原抓点，复测失败并停止下降。此检查是位置和类别约束，在遮挡或两个同类目标无法分离时仍需现场验证。

地面和搁台同框时，可在配置中使用 `support_regions_mm`，每项为底座坐标中的 `[[xmin,ymin,zmin],[xmax,ymax,zmax]]`。每个区域独立拟合支承面，XY 范围须分离，Z 范围只包住对应支承面。悬停复测保持原区域。同一检测框跨区罩住多个物体时，仍不能单独确认其中一件。上下重叠的搁台需使用不同观测工位或配置。

下降前比较悬停三帧的所有两两组合。物块表面点和计划抓取 TCP 分别最多相差 3 mm，夹爪 Roll 最多相差 5°。每帧的表面点和抓取 TCP 还必须在初次估计的 8 mm 内，Roll 在 10° 内。通过后执行最后一帧的方案。5° 是初始工程门槛，待实装复测。

## 实机关节轨迹的离线扫掠审计

capture_joints.py 可在机械臂由独立入口点动时录制 T=105 反馈。采集器仅查询，不下发运动。串口不能与其他串口进程共享，需使用设备允许的独立点动方式。连续采集间隔超过 0.15 s 或相邻角度跳变超过 0.10 rad 时，审计判定该段采样不足。轨迹也可以由其他采集器生成，CSV 表头为 time_s,b,s,e,t,r,g。

```powershell
python capture_joints.py COM5 bin-trace.csv --seconds 15
python audit_sweep.py config.json measured-workspace.json bin-trace.csv sweep-report.json
```

measured-workspace.json 要求 padding_mm、jaw_sign、jaw_offset_rad、obstacles 和 wrist_envelopes。两个几何数组每项含 name、center_mm 和 size_mm（完整尺寸）。障碍箱在底座系，腕部箱在 link5 系，使用包住实物的轴对齐外包盒。夹爪角应先核对真实 g 到 Xacro 铰链的映射。审计使用官方连杆和分段原装夹爪，保留模型中相邻连杆排除规则。此离线检查覆盖录制轨迹的离散采样和区间插值。带物轨迹可在 workspace 中增加 payload_envelope，格式与腕部外包盒相同，中心和完整尺寸在 link5 系，以毫米计。只输入从确认持物到释放前的轨迹，整个区间都按夹持物随腕部运动检查。样本外包盒与场地障碍接触会单独报 payload_collision，夹爪与样本的接触不参加此检查。报告记录输入文件指纹。真实夹持滑移、实际夹爪与样本接触及未录制的运动仍需独立验证。


## 固件坐标与官方模型 TCP 对照

capture_joints.py 现在也记录 T=105 的 x、y、z。用独立的两个录制文件检查固件 TCP，分别覆盖至少六个静止或缓慢点动的不同腕姿态，包含明显的 Pitch 和 Roll 变化。旧轨迹 CSV 仍能用于扫掠审计，TCP 对照需要新列。

```powershell
python validate_tcp.py config.json tcp-fit.csv tcp-check.csv tcp-report.json
```

报告比较官方 hand_tcp 的正解和固件报告位置，并诊断常量腕部工具偏移与基座平移能否解释差异。独立检查必须包含未在拟合组出现的姿态。诊断偏移不会自动应用。若配置 tcp_compatibility_report，腕部运动会检查报告通过且官方模型与关节映射仍匹配。即使固件反馈与模型一致，仍需用实体探针或标定靶独立检查实际工具中心。

### 实体探针独立验证

validate_tcp.py 可追加 --probe probe.csv。CSV 表头为 target_id,b,s,e,t,r,x,y,z,tip_x_mm,tip_y_mm,tip_z_mm。后面三个坐标是独立测量的底座系触点，不能填回机械臂自身报告。探针尖端需代表实际抓取工具中心；若是加装标定针，要先确定尖端到实际 TCP 的关系。至少使用三个空间分散且非共线的参考点，每点独立重复触碰两次。脚本只读 CSV，不控制探针运动。

```powershell
python validate_tcp.py config.json tcp-fit.csv tcp-check.csv tcp-report.json --probe probe.csv
```

固件报告和模型预测都须落在独立实测参考的误差门限内。若配置 require_physical_tcp 为 true，缺少探针通过报告时腕部运动拒绝执行。即使未启用这一强制项，明确未通过的探针报告也会阻止动作。合成测试覆盖固件与模型一致而实体参考偏离时拒绝。

## 上车前离线证据预检

deployment_preflight.py 不连接硬件。它统一检查部署模式、腕部定位报告、实体 TCP、夹爪反馈报告、目标录帧验收、料仓复核配置和带物扫掠报告。配置、模型、相机射线、检测权重、场地或轨迹发生变化后，按检查结果重新生成相应报告。

料仓未定型时，可加 `--grasp-only` 做抓取与抬升的台架预检。该模式仍核对实测定位、实体 TCP、夹爪反馈、双模型录帧验收和带物扫掠，报告标记为 `bench_grasp_only`。

台架抓取的 JSONL 可直接做带物扫掠审计。试提和全程抬升分别检查，避免跨越停顿插值。预检使用同一份 JSONL 和生成的扫掠报告。审计覆盖有反馈的区间，第一条反馈前的动作和实物未建模部分仍需现场检查。

扫掠审计要求本次命令有到位标记 `motion_complete`，且首末反馈各自距命令边界不超过 0.15 秒。较早生成、缺少到位标记的 JSONL 需重新录制才可用于此项验收。

```powershell
python audit_sweep.py config.json measured-workspace.json bench-pick-001.jsonl sweep-report.json --motion-stage proof_lift --motion-stage full_lift
```

```powershell
python deployment_preflight.py config.json --camera orbbec --rgbd scene.npz --weights $detector --grasp-checkpoint $ggcnnWeights --target-report field-report.json --sweep-report sweep-report.json --workspace measured-workspace.json --trace carry.csv --report preflight.json
```

部署预检逐帧检查目标报告。验收录帧至少要覆盖有可抓特殊人造物和无目标两种情况，每帧只计一次。现场采集时加入普通木块与特殊人造物同框、多个特殊人造物同框，以及只有木块或岩石的场景。

GG-CNN 模式还要指定 --backend ggcnn 与 --grasp-checkpoint，并使用新版本录帧验收生成的报告。预检通过表示所给离线证据一致。实车报告必须由对应硬件、场地与独立采样产生，预检不判定报告来源，也不证明整车完整比赛流程。

## 整车任务证据与计分复盘

run.py 的 --sample-id 可将上层分配的样本 ID 与识别类别写入终态日志，需同时开启语义确认、入仓和日志。上层需独立确认目标身份与去重；仅执行到夹爪打开的 released_unverified 不对应入仓证据。

mission_evidence.py 离线汇总两次出发，每次按 600 秒内的离场、正确目标入仓、停稳返航且有保留样本、完整落在卸载区并停稳的事件计分。场景 JSON 提供 allowed_classes 和实测地图坐标 unload_zone_m=[x0,y0,x1,y1]。每条 JSONL 事件带 time_s 与 event。departed 需 base_clear，stored 需 sample_id、target_class、deposit_depth_confirmed，returned 需 base_inside、stopped 与 retained_samples，unloaded 需 sample_id、settled 和 footprint_m。落地区域用物体完整平面包络，不能用单个中心点。

原整车工程的 /grasp/result 对 SUCCESS 增加已收集数量，UNLOAD_COMPLETE 结束任务。原卸载完成由定时器触发，抓取超时后会恢复探索。当前没有将 M3-S 日志自动发布到这些话题，也未改动原仓库。集成前须加入底盘互锁、机械臂收回证明、实测卸载证据和可靠的失败恢复，避免用开爪或超时替代这些确认。

## 收臂与底盘交接证据

在 storage.stow 中提供独立调试过的 commissioned、from_release、from_view 和 joint_bounds_rad。路径是 [x_mm,y_mm,z_mm,pitch_rad,roll_rad] 的列表；开启独立复核位时两条路径均必填。关节边界是两行六列的原始固件角度下上界，顺序 b,s,e,t,r,g，单位 rad。应覆盖已验证的真实运输姿态，并检查空载开爪时整臂、相机、线束和车体的扫掠间隙。未配置 stow 时不执行收臂，终态 arm_stowed 为 false。

释放或复核后按实际所在位置选择收臂路线。最后三个反馈样本都必须在关节包络内，且关节不再明显变化，终态才记录 arm_stowed: true。它是供上层使用的一次性记录。底盘仍需独立实时互锁，不能仅凭这条旧日志启动。离线部署预检检查此配置是否完整。

## 底盘 ROS 2 实时互锁入口

原车 nav_to_manual_mux.py 的输出可重映射为 /manual_control_ungated，由 ros_transport_interlock.py 统一输出 /manual_control 给 motion_solver。上线时所有底盘速度来源都必须经过门控，确认没有并行发布者，并配置下游失联停车看门狗。不要与原输出直接并联。

门控订阅 /m3s/transport_feedback 的 JSON String。消息含本机 monotonic_s、busy 和六个原始关节 b,s,e,t,r,g。发布方须为同一车载主机的机械臂串口唯一所有者，提供新的真实反馈，动作前先撤销许可。门控需要三个覆盖至少 0.10 s 的稳定姿态样本，反馈超过 0.25 s 或底盘请求超过 0.20 s 均输出空 ManualControl。许可恢复后需要新的底盘请求，旧命令不会被恢复。

目前 ROS 2 接口按原工程消息类型实现，未在 ROS 2 或车载硬件运行。持续反馈由独立的收臂反馈进程提供。协调器已实现进程接力，原车速度源重映射仍待实车联调。缺反馈时默认停车。

## 机械臂动作前的底盘停车握手

上车联调时设置 require_base_interlock 为 true。执行入口在首个机械臂动作前发送带唯一令牌的 /m3s/base_lock_request。ROS 2 门控撤销底盘速度并锁存禁止移动后，返回相同令牌的 /m3s/base_lock_ack。机械臂端在收到确认后仍需观察至少 0.30 s 的连续静止里程计，再进入动作。默认里程计为原真车导航配置的 /localization/lio_odometry，可通过 odom_topic 替换。握手超时不发机械臂动作。

抓取结束不会自行恢复底盘。成功确认入仓并收臂的终态日志带锁令牌，可供上层启动独立只读反馈进程。不要把节点重启当作解锁流程。实车仍需确认唯一速度路径、里程计时延和底盘失联看门狗。


### 解锁与串口交接

先等待抓取进程完全退出。上层核对任务状态后，使用同一配置和该次新日志启动 python ros_stowed_heartbeat.py config.json --motion-log bench-store-001.jsonl @pickModels。该进程独占串口并只发反馈查询，要求日志记录已确认入仓、已收臂和本次锁令牌。门控还要求锁车以后收到新的稳定关节反馈，才接受匹配的解锁请求。解锁清空历史速度，恢复移动需要之后的新反馈和新速度请求。停止反馈进程后许可会过期。下一次抓取前需停止反馈进程并交还串口。失败或证据不足保持锁车，不能使用旧成功日志解锁。该流程尚未在车载 ROS 2 执行。


### 原任务节点与 M3-S 协调器

ros_cycle_supervisor.py 使用原车 /mission/state 和 /grasp/trigger，替代原四轴抓取节点作为 /grasp/result 的唯一抓取结果发布者。启动时使用只读初始收臂反馈，收到本轮触发后等待反馈进程退出才启动抓取。抓取子进程结束后，根据新日志切换到独立反馈进程。只有当前任务仍在 GRASPING 且底盘门控发来与该次令牌匹配的新鲜 transport_ready 时，才发布 SUCCESS。相同抓取阶段只消费一次触发。乱序触发最多缓冲 0.5 s。失败报告到 /m3s/cycle_status，不向原任务节点发送会立即恢复探索的 FAILED。

```text
python ros_cycle_supervisor.py config.json logs --camera orbbec @pickModels
```

进程终止只能停止软件继续下发，不能保证已进入固件的动作立即中止。需在实车测试底盘互锁、独立停车与 ROS 话题接线。未实现 M3-S 卸载控制，不会发布 UNLOAD_COMPLETE。

## 卸载区的独立深度验收

unload_depth.py 使用固定且已测定地图位姿的相机。给定卸载区地图坐标 unload_zone_m=[x0,y0,x1,y1] 和该区域地面高度 floor_z_mm。用同一个深度流在卸载前采一帧，释放停稳后采三帧。脚本比较可见高度的变化，要求新物体数量正确、连续稳定、边缘附近有足够深度覆盖，并给可见外包框每侧增加 10 mm 余量后仍在区内。

```powershell
python unload_depth.py scenario.json map_from_depth.npy before.npz after1.npz after2.npz after3.npz --expected 2 --report unload.json
python simulate_unload_evidence.py
```

物理验证从空区观测开始，让木块在 MuJoCo 里自由落地后从模拟深度复核。两件均在区内、跨边线及少件三种场景均得到预期结论。工具不能识别样本身份或证明被遮挡的尺寸，也未连接实际卸载执行器及 ROS 完成话题。叠放、遮挡及场地不平需实测验证。

## 对照仿真和真机轨迹

仿真用 `--trace` 生成带采样时刻的 NPZ，真机用 `capture_joints.py` 记录 CSV。分别标出同一动作的开始和结束时刻，以下秒数仅为参数格式示例。

```powershell
python compare_motion.py config.json sim_output/timed-storage-ggcnn_0.npz bench.csv --sim-window 0.1 5.0 --hardware-window 2.1 7.4
```

脚本按配置转换真机关节符号和零位，在共同经过的时间段内计算角度和速度误差，并单独报告动作耗时差与结束姿态差。它不会拉伸时间轴。默认拒绝比较窗口内超过 0.15 秒的采样空档。

自动抓取开启 `--motion-log` 后，也可直接把生成的 JSONL 传给 `compare_motion.py`。其中 `motion_sample` 标记动作中与到位时的 T=105 反馈，`stage` 可用来标记下降或抬升区间。采样时刻以请求与回包的主机时间中点估计，间隔受串口和固件响应影响。

对带阶段标记的仿真轨迹和新版抓取 JSONL，可用 `--stage descend`、`--stage proof_lift` 或 `--stage full_lift` 代替两组手动时间。真机的对照区间取该阶段第一条到最后一条反馈，发送指令到首条反馈之间的延迟不计入该区间。


### 2026-11-12：打印棋盘手眼采集入口

使用打印手册的8x6方格（7x5内角点），默认每格20mm，以尺量实测边长为准。连接Gemini335L及COM6后运行 `python debug.py chess`；启动串口可能使机械臂归位，先清空运动范围。

在棋盘外白边标记固定的一角。普通对称棋盘不能唯一辨别180°反序，因此每个样本保存前，在原始RGB窗口点击该标记角旁最近的最外侧内角点；点击只选定检测角点的排序，不用鼠标坐标替代亚像素角点。看到ORIGIN标记后，机械臂静止时按S保存。每次保存会清除原点选择，下一姿态重新点击。F为求解组，C为验证组，Q退出；建议20组求解、5组独立验证。板子/底座不动，相机支架不松动，只调关节姿态。

运行 `python debug.py solve` 离线求解最新完整chess-*会话。复用手眼算法和独立验证门限，不自动安装外参或启用运动。非20mm方格时直接用 `python chess_capture.py --square-mm 实测边长`，需使用debug.py相同的依赖环境。

`python test_chess_handeye.py -v` 覆盖棋盘检测、四种反序的物理原点恢复、缺少原点拒绝及已知外参的完整15+5合成会话回归；已通过。这些不代表现场精度已通过。


### 14.93mm Tag候选定位预览（2026-11-12）

`python debug.py preview14` 在Gemini335L的YOLO/GG-CNN图像上显示选定表面的候选基座XYZ。默认明确加载 `calibration/tag-1794451675210315500/solve-1794452125588083000/report.json` 中的深度相机到link5外参；不修改报告、不将commissioned设为true、不发送运动或力矩指令。查询COM6实际角度，打开串口仍可能触发固件归位，需清空范围并关闭其他串口程序。

点击RGB或深度窗口选表面点，运动/无动力或深度无效时隐藏坐标；Q退出。此入口不做抓取执行，不将14.93mm视为抓取误差上限。`python preview_candidate.py --bias-mm X Y Z` 可在候选预览中添加基座XYZ偏置，仅用于已测量的固定偏差试验，不改变机器人配置或原外参。偏置不能修复随姿态变化的误差。

离线使用 `--frame ...npz --headless --pixel U V --output ...png` 不打开相机或机械臂。已在真实保存帧验证模型加载、坐标投影和显示输出，位姿与偏置的mm单位另有数值检查；未执行实机运动。


### 真实木块台架与原始RGB输入（2026-11-12）

`python debug.py preview14` 台架预览默认选择 `wood_block`，比赛入口仍默认 `artificial_object`；可直接用 `python preview_candidate.py --target-class artificial_object` 查看比赛类。原权重包含类别名称不等于已适配当前实物，用户提供的4张手机木块参考图在0.35阈值下均未检出，结果保存在 `candidate_preview/reference_photos/report.json`，这不是正式测试集精度。

Gemini335L具备原始RGB内参及深度到RGB外参时，YOLO改用完整原始RGB。检测框通过有效深度点投影回深度光栅，包含RGB遮挡近表面过滤；GG-CNN继续使用实际深度。左侧原始RGB采用等比例显示，右侧深度空洞保持无效，不因彩色图完整就伪造深度。点击左侧RGB仅在附近3像素内存在实际投影深度时显示XYZ，否则提示无有效深度。抓取矩形画在右侧深度图，左侧可显示对应中心。

新的 `python debug.py data` 只打开相机，不加载YOLO/GG-CNN、不打开机械臂串口。S保存，Q退出。目录 `datasets/competition/raw/session-*/<时间编号>/` 包含 `original_rgb.png`、`aligned_rgb.png`、`rgbd.npz`、`capture.json`。优先在原始RGB上标YOLO框；原始/对齐图不可混用坐标。真实深度及工厂标定信息保存在NPZ。所有新帧标为unlabelled，未检出目标不自动当成负样本，YOLO框不作为GG-CNN抓取真值。同一场景的连续帧不能随机混入训练和验证；可通过 `python data_capture.py --scene 场景编号` 指定跨次采集的场景分组。

已通过投影单位、深度随距离变化、无效深度、RGB遮挡及原始RGB不被深度洞遮盖的测试，并用已保存的真实RGB-D帧离线运行模型；未自动运动或宣称实机抓取通过。


### 三种物块首轮100张采集计划（2026-11-12）

用户确认首轮计划为三种物块各20张、混放30张、无目标背景10张，每一种新摆放只拍1张，不必使用Wi-Fi移动机械臂。固定相机，直接移动/旋转/翻面物块，改变背景，不要原地重复保存。

`python debug.py data` 新界面显示中文计划、各批次数量、总进度和下一步提示。按1/2/3选择单物块批次、M混放、B背景，S保存，Q或窗口关闭键退出；达到每批目标后提示换批次，不自动假定物块已换。默认每张照片视为操作员按要求更换过的摆放组；同一摆放需要重复保存或跨次同场景分组时，用显式 `--scene` 标记。保存的批次只是人工采集意图，contains_target仍为null，仍需审核和画框，背景批次不会自动作为已验证负样本。

此更新只修改源码，未替用户启动/重启程序。重启需用户先关闭旧版窗口再运行命令。已修复旧循环反复创建窗口导致点击关闭后重现的问题；正常窗口关闭会结束相机上下文。测试不打开真实相机。若旧版已卡死，可在PowerShell按data_capture.py完整工程路径筛选Python进程后强制结束。
# 已采集图像的目标框标注

五物块远近补采：`python debug.py data5`，使用当前确认的预设（含轻度去噪），原生MJPG软件解码，左RGB/右深度，显示180°；不加载YOLO或GG-CNN，不打开机械臂串口。`F`远景、`N`近景，`1`～`5`选择对应物块，`M`混放、`B`背景，`S`换摆放停稳后拍一张，`Q`退出。每种物块远25/近10，混放远10/近5、背景远5/近5，总计远140/近60=200。五种物块编号固定，混放需要包含不同组合。远近通过改变真实相机/物块距离获得，不用数字缩放替代；深度盲区中的RGB可补YOLO，但不算抓取深度验证。窗口显示全部批次计数和下一步提示，批次满后拒绝原地继续堆图。重新启动自动续采未完成会话，按实际已完成capture.json重建计数；预设变化时拒绝混入同一会话。每张保留原始RGB、深度、射线/工厂RGB外参、预设SHA和远近批次。采完仍需 `label`画框、`prepare`导出再`train`，训练检查预设指纹，旧100张不能伪装成新配置。

预设链路联调：`python preset_preview.py` 加载工程内冻结预设、新 YOLO 和 Cornell GG-CNN，左侧原生 RGB 黄色检测框、右侧原生深度绿色抓取矩形；两幅图单独旋转180°、文字正向，推理坐标保持原生。有效候选显示相机 XYZ。`python preset_preview.py --port COM6` 增加前后五关节反馈一致性检查，使用当前14.93mm残差候选手眼外参显示候选底座 XYZ；串口可能导致固件重启回初始姿态，不发送主动运动指令，不是部署验收。没有有效深度、GG-CNN候选或关节反馈时不会编造XYZ。`S`保存原始RGB、深度、工厂射线及RGB外参、可用关节角和结果，`Q`退出。坐标变换和无效深度行为有离线测试；当前预设下实时效果仍需现场验证。远距离小目标尚未补采，YOLO漏检时不会调用其它目标的候选来冒充成功。

完整图像推理回放：`python evaluate_preview.py`，读取当前训练导出的20张验证图对应原始 RGB-D，运行 YOLO + GG-CNN，报告 `candidate_preview/heldout-pipeline.json`。固定 YOLO 置信度0.35、RGB框 IoU0.5 一对一匹配；与训练验证输出的最优F1工作点精确率不是同一个统计口径。2026-11-12结果：24个匹配、2个误检、1个漏检；18张有目标图均未生成GG-CNN抓取候选。不能将 YOLO 验证 mAP 或零错误运动解读为抓取链路验收通过。脚本不打开硬件。

原始深度诊断：`python debug.py depth`，仅打开 Gemini，相机原生深度窗口拖框选择目标区域，显示全图与区域内有效深度比例；`S` 保存未填补的 RGB-D、深度可视化和统计到 `depth_diagnostics/<时间戳>`，`Q` 退出。不打开串口、不改相机设置。先关闭其他占用相机的程序。离线检查整次采集：`python depth_health.py --session datasets/competition/raw/session-1794472596073647000`，结果 `depth_diagnostics/session-health.json`。诊断图会按帧深度分位数调整颜色，颜色不可跨帧当作固定距离比较，距离见统计文件。

训练完成后的相机预览：`python debug.py trained`，加载 `latest-training.json` 的最佳权重与既有 GG-CNN，不打开机械臂串口。需要查看候选底座 XYZ 时用 `python debug.py trained14`；该命令会打开 COM6 获取反馈，可能触发控制器重启回初始姿态，需清空运动范围；14.93 mm 外参仍未通过运动部署验证。

2026-11-12 离线核验：新权重 SHA256 `7f0515d3dbf82c559fa52ecd16ac896856b76b999210a4f12f730524625467a8`，回放图与推理记录在 `candidate_preview/trained-offline.png` / `.json`。该帧识别到人工目标，但因目标区域深度空洞拒绝 GG-CNN；本次100帧全图有效深度中位数约4.9%（1.3%～18.1%），不能据此声称抓取链路或实机部署已验证。RGB 标签可用于本轮 YOLO 训练；抓取推理需进一步排查深度质量。

标注完成后运行 `python debug.py prepare`，严格检查审核状态、图像大小、框与 YOLO 标签的一致性，按场景分组并按采集批次划分约 80% 训练、20% 验证；完全相同图像也归入同组。原始数据不修改，导出到 `datasets/competition/yolo/export-<时间戳>`。小样本验证不代表比赛场景的独立测试。

随后运行 `python debug.py train`，默认使用已有 `D:/Lab/挑战杯/LunarRover_HUST-master/weights/best.pt` 微调单类 `artificial_object`，自动优先使用 CUDA 0、不可用时使用 CPU；640 像素、batch=4、最多100轮、早停25轮。每次创建独立的 `runs/targets/targets-<时间戳>`，不会覆盖旧权重。成功后终端显示 `BEST WEIGHTS`，并保存 `datasets/competition/latest-training.json`。只训练 YOLO，不训练 GG-CNN，不自动切换实机预览权重。

RTX4060 笔记本首次运行 `python setup_gpu.py`，将 PyTorch2.7.1 / torchvision0.22.1 CUDA12.6 安装到项目 `.cudadeps`，不替换全局环境和原 CPU 依赖。安装后自动验证 CUDA 矩阵运算、GPU NMS、已有权重加载；没有通过则不会提示 READY。该脚本只安装与检查，不启动训练；下载文件较大。版本配对见 [PyTorch 官方安装列表](https://pytorch.org/get-started/previous-versions/)。

仅检查输入与模型加载：`python train_targets.py --check`（不训练）。参数调整示例：`python train_targets.py --epochs 100 --batch 4 --device cpu`。Windows 使用主入口保护和 workers=0；训练接口参见 [Ultralytics 官方说明](https://docs.ultralytics.com/modes/train/)。

在项目目录运行 `python debug.py label`，打开最近一次已保存的数据集，不连接相机或机械臂。鼠标拖框标出每个目标的完整可见范围；三种待抓物块统一为 `artificial_object`。混放图中每个目标分别画框，其他物品不标。

- `S`：保存当前所有框并下一张；无框时不会自动当作背景。
- `B`：确认本张没有目标，保存空标签并下一张。
- `Backspace`：撤销最后一个框；`R`：清空当前框。
- `A` / `D`：上一张 / 下一张（不保存当前改动）；`Q`：退出。

再次运行自动从第一张未审核图像继续。每个采集子目录保存 `annotation.json`、对应原始 RGB 的 YOLO `.txt`，并更新 `capture.json`；原始图像和深度保留。

