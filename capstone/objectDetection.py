# from ultralytics import YOLO
# import os
# #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# # 如果上面的不行，试试下面这个专门针对 objc 冲突的
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# # Configure the tracking parameters and run the tracker
# model = YOLO("yolo26n.pt")
# results = model.track(source="31-3.MOV", conf=0.1, iou=0.7,stream=True)
# for frame_idx, r in enumerate(results):
#     # 1. 获取检测框的坐标 (Boxes)
#     # r.boxes.xyxy 是 [xmin, ymin, xmax, ymax] 格式
#     boxes = r.boxes.xyxy.cpu().numpy()
#
#     # 2. 获取类别索引 (Class IDs)
#     clss = r.boxes.cls.cpu().numpy()
#
#     # 3. 获取置信度 (Confidences)
#     confs = r.boxes.conf.cpu().numpy()
#
#     # 4. 获取追踪 ID (Track IDs) - 只有 track 模式才有
#     if r.boxes.id is not None:
#         track_ids = r.boxes.id.int().cpu().tolist()
#     else:
#         track_ids = []
#
#     # 打印当前帧信息
#     print(f"--- 第 {frame_idx} 帧 ---")
#     for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):
#         name = model.names[int(cls)]  # 将索引转换为类别名称，如 'person'
#         print(f"ID: {track_id}, 类别: {name}, 置信度: {conf:.2f}, 坐标: {box}")


import os
import cv2  # 需要安装 opencv-python
from ultralytics import YOLO

# 1. 加载模型 (n 代表 nano，最适合实时处理)
model = YOLO("yolo26n.pt")

# 2. 运行追踪，source=0 表示调用摄像头
# show=True 会自动弹出一个窗口显示画面
# stream=True 确保内存不会因为视频流而爆炸
results = model.track(source="0", show=True, stream=True,conf=0.3)

print("正在启动摄像头... 按 'q' 键退出预览窗口")

for r in results:
    # --- 获取每一帧的检测信息 ---
    #print(r)
    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()  # 坐标
        track_ids = r.boxes.id.int().cpu().tolist()  # 追踪ID
        clss = r.boxes.cls.cpu().tolist()  # 类别索引

        for box, track_id, cls in zip(boxes, track_ids, clss):
            name = model.names[int(cls)]
            x1, y1, x2, y2 = box
            # 打印当前帧检测到的目标信息
            print(f"detect到: {name} (ID: {track_id}) | position: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    # 注意：如果你使用了 show=True，ultralytics 内部会处理窗口显示。
    # 如果你想手动控制退出，可以在这里加入逻辑。