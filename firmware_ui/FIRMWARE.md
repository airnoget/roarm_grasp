# v11固件

源码目录：`.vendor/wifi-debug/RoArm-M3_example/`。网页由`index.html`与`control_protocol.js`拼接生成，运行`python firmware_ui/build_header.py`更新固件头文件。

原台架使用Arduino ESP32 core 3.0.7，FQBN：

```text
esp32:esp32:esp32:PartitionScheme=default,PSRAM=enabled,FlashSize=4M,FlashMode=dio,FlashFreq=80
```

第三方库按固件中的include和上游RoArm-M3示例安装；不含本机Arduino缓存。上游来源：[Waveshare RoArm-M3](https://github.com/waveshareteam/RoArm-M3)、[roarm_ws](https://github.com/waveshareteam/roarm_ws)。第三方组件遵守各自上游许可；本仓库不重新授权这些组件。

`releases/roarm-m3-v11-base360-pitch38.bin`为ESP32应用区镜像，写入偏移0x10000。构建及哈希记录见`build_record_v11.json`。原记录中的COM6、MAC、路径是构建台架记录，不是通用设备配置。

烧录会重启控制器，恢复舵机电源时会按开机姿态初始化。更换机械结构时需要先核对其可达范围与线缆。v11初始俯仰-38度、底座网页范围±179.9度；没有将ID11切换为轮式连续旋转模式。
