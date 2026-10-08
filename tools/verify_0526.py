#!/usr/bin/env python3
"""0526.onnx 冒烟测试：按同济 yolov5.cpp 的解析逻辑（letterbox + 22 列布局）跑推理并画图。
布局（深大 README / 同济 yolov5.cpp 一致）：
  0-7   四个关键点 (x,y)x4，左上角起逆时针
  8     objectness（需 sigmoid）
  9-12  颜色 argmax（**实测顺序，非 README 文字顺序**：蓝 红 灰 紫）
  13-21 数字 argmax：G/1/2/3/4/5/O/Bs/Bb
"""
import sys
import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL = os.environ.get(
    "MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "0526.onnx"),
)
COLORS = ["blue", "red", "gray", "purple"]  # 实测：蓝=0 红=1 灰=2 紫=3
NUMS = ["G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"]
CONF_TH = 0.50
NMS_TH = 0.45


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out_meta = sess.get_outputs()[0]
print(f"模型: {inp.name} {inp.shape} ({inp.type}) -> {out_meta.name} {out_meta.shape} ({out_meta.type})")

for path in sys.argv[1:]:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    # ---- letterbox（同济 yolov5 同款：等比缩放到 640，左上对齐补黑） ----
    scale = min(640 / h, 640 / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))

    x = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, /255
    x = x.transpose(2, 0, 1)[None]                      # NCHW
    if "float16" in inp.type:
        x = x.astype(np.float16)

    out = sess.run(None, {inp.name: x})[0][0].astype(np.float32)   # (25200, 22)

    conf = sigmoid(out[:, 8])
    cand = np.where(conf > CONF_TH)[0]
    boxes, scores, dets = [], [], []
    for i in cand:
        row = out[i]
        c = int(np.argmax(row[9:13]))
        n = int(np.argmax(row[13:22]))
        pts = [  # 同济/深大同款重排：(0,1)(6,7)(4,5)(2,3) -> TL,TR,BR,BL
            (row[0] / scale, row[1] / scale),
            (row[6] / scale, row[7] / scale),
            (row[4] / scale, row[5] / scale),
            (row[2] / scale, row[3] / scale),
        ]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        boxes.append([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)])
        scores.append(float(conf[i]))
        dets.append((pts, c, n, float(conf[i])))

    keep = []
    if boxes:
        keep = np.array(cv2.dnn.NMSBoxes(boxes, scores, CONF_TH, NMS_TH)).flatten().tolist()

    print(f"\n{path}  ({w}x{h})  原始候选 {len(cand)} -> NMS 后 {len(keep)}")
    vis = img.copy()
    for k in keep:
        pts, c, n, cf = dets[int(k)]
        poly = np.array([[int(p[0]), int(p[1])] for p in pts])
        cv2.polylines(vis, [poly], True, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
        label = f"{COLORS[c]} {NUMS[n]} {cf:.2f}"
        cv2.putText(vis, label, (poly[:, 0].min(), max(30, poly[:, 1].min() - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        print(f"   {label:22s} 角点={[ (round(p[0]),round(p[1])) for p in pts ]}")

    dst = path.rsplit(".", 1)[0] + "_det.png"
    cv2.imwrite(dst, vis)
    print(f"   -> {dst}")
