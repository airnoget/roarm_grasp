# 目标 relative_xy 接口与下位机 world_xy 解算

2026-11-16。上位机已输出JSON文件，串口/网络运输由后续适配；本版本不向下位机发送控制命令。

## 获取目标

运行 `python debug.py approach`（或locate），锁定目标后读取工程根目录 exploration_output/relative-target.json。多目标默认最近，Tab可切换，机械臂抓取运动期间保留选中ID。目标当前无有效深度或机械臂运动时输出valid=false及relative_xy=null；接收端必须忽略旧值。没有新文件更新也不能无限沿用最后一次数据。当前ID在固定机械臂底座参考系内关联，车辆移动后跨位置身份保持尚未接入SLAM，不能当作全场永久编号。

示例：
```json
{"schema":1,"target_id":2,"stamp_unix_s":1794790000.0,"frame_id":"arm_base","units":"m","relative_xy":[0.4,0.1],"valid":true,"measurement":"top_surface_center","world_xy":null}
```

relative_xy来自当前手眼+FK得到的物块顶面中心，单位米；不是相机坐标，也不是距离增量，更不是底盘速度。Z和完整抓取几何仍保存在本工程result/抓取规划中，不输出给此二维接口。编号相同的多帧是同一目标的更新，不是新增任务。机械臂自身运动不会直接改变静止物块的底座坐标，但测量误差仍存在。

## 参考系必须核对

arm_base是机械臂基座原点，按本工程实际FK轴定义验证前后左右；预期X向前、Y向左。车辆SLAM的base_link原点通常与它不同，不能直接把relative_xy加到SLAM位置上。装车后量取arm_base相对底盘的平面安装偏移(tx,ty)及偏航alpha；若安装有明显倾斜或车辆有坡度，应使用完整三维变换，二维接口不足以补偿高度耦合，届时扩展输出XYZ。

## 平地二维解算

下位机/车载计算端取得与观测时间对应的SLAM底盘位置(world_x,world_y)和航向theta，弧度、逆时针为正。仅有world_xy而没有航向不能完成解算。

先把机械臂基座相对坐标(rx,ry)换到底盘：
```
bx = cos(alpha)*rx - sin(alpha)*ry + tx
by = sin(alpha)*rx + cos(alpha)*ry + ty
target_world_x = world_x + cos(theta)*bx - sin(theta)*by
target_world_y = world_y + sin(theta)*bx + cos(theta)*by
```
例如安装无偏移，车位(2,3)、航向90度、relative_xy=(0.4,0.1)，目标world_xy=(1.9,3.4)米。

SLAM帧应明确map或odom，不混用。接收端按target_id保存目标世界坐标和观测时间；使用观测时刻的位姿而非收到包时的最新位姿。跨机器需同步时钟；当前stamp_unix_s为上位机处理时刻，尚不是硬件曝光同步时间，因此首版采用停车观测再解算，移动观测需接入硬件帧时间、关节时间及SLAM插值。换到另一台机器时需带会话ID，避免重启后的编号与旧任务混淆。

当前文件作为最新状态接口，后续串口/CAN/网络封包需保持这些语义，补充会话ID、序号、校验、确认及有效期。JSON读取失败重读下一帧；valid=false或数据停止更新时停车/重观测，不依据缓存继续追旧点。world_xy目标不是可直接驶入的停靠点，仍需避障和机械臂工作区规划。
