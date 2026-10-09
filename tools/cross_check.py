#!/usr/bin/env python3
"""0526 检测对拍：Python(onnxruntime) 参考实现 vs C++(TensorRT) 输出逐框比对。

为什么：C++ 解码逻辑（关键点重排/颜色编号 argmax/NMS）必须和经过验证的 python
参考实现数字对得上，才能说"没有写错"。两边跑同一个模型（同一份 0526.onnx / 由它
构建的 engine），浮点路径不同（TRT fp16 vs onnxruntime），所以允许小差。

用法（C++ 在 Jetson，Python 在任意装了 onnxruntime 的机器）：
  # Jetson：
  cd ~/rm_vision && for f in a.jpg b.png; do ./build/yolo_test $f 2>&1; done > /tmp/cpp.txt
  # 对拍机（同样的图片文件 + 模型）：
  MODEL=/path/to/0526.onnx python3 tools/cross_check.py /tmp/cpp.txt a.jpg b.png

判据（每个检出按"中心就近"配对后）：
  - 颜色一致（python 的 gray = C++ 的 extinguish，其余同名）
  - 编号一致（G↔sentry / 1..5↔one..five / O↔outpost / Bs↔base+small / Bb↔base+big）
  - 置信度差 ≤ 0.03
  - 四角点坐标最大差 ≤ 2.0 px
"某一侧多出没配上的检出"会先按低阈值复跑 python 侧确认：
  若它在低阈值下存在且 conf 与 0.50 只差 ≤ 0.03 → 判为**阈值边缘**（两边浮点路径
  不同，一取一舍），只警告不算失败（实测案例：C++ 0.504 vs python 0.495）
任何一项真超差 → 退出码 1
"""
import os
import re
import sys

import cv2
import numpy as np
import onnxruntime as ort

CONF_TH, NMS_TH = 0.50, 0.45
PY_COLORS = ["blue", "red", "gray", "purple"]
PY_NUMS = ["G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"]
# python 编号 -> (C++ name, C++ type 或 None 表示不看型别)
NUM_MAP = {
    "G": ("sentry", None), "1": ("one", None), "2": ("two", None), "3": ("three", None),
    "4": ("four", None), "5": ("five", None), "O": ("outpost", None),
    "Bs": ("base", "small"), "Bb": ("base", "big"),
}
CONF_TOL, PT_TOL = 0.03, 2.0

DET_RE = re.compile(
    r"^(\w+)\s+(\w+)\s+(\w+)\s+conf=([\d.]+)\s+color_id=(\d+)\s+num_id=(\d+)\s+"
    r"ratio=([\d.]+)\s+rect_err=([\d.]+)")
PTS_RE = re.compile(
    r"TL=\(([-\d.]+),([-\d.]+)\)\s+TR=\(([-\d.]+),([-\d.]+)\)\s+"
    r"BR=\(([-\d.]+),([-\d.]+)\)\s+BL=\(([-\d.]+),([-\d.]+)\)")
IN_RE = re.compile(r"输入 (\S+)\s+(\d+)x(\d+)")


def parse_cpp(path):
    """解析 yolo_test 的 stdout -> {basename: [det, ...]}"""
    out, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = IN_RE.search(line)
        if m:
            cur = os.path.basename(m.group(1))
            out.setdefault(cur, [])
            continue
        if cur is None:
            continue
        m = DET_RE.match(line.strip())
        if m:
            cur_det = {"color": m.group(1), "name": m.group(2), "type": m.group(3),
                       "conf": float(m.group(4)), "color_id": int(m.group(5)),
                       "num_id": int(m.group(6))}
            out[cur].append(cur_det)
            continue
        m = PTS_RE.search(line)
        if m and out[cur]:
            g = [float(x) for x in m.groups()]
            out[cur][-1]["pts"] = [(g[0], g[1]), (g[2], g[3]), (g[4], g[5]), (g[6], g[7])]
    return out


def reference(sess, path, conf_th=CONF_TH):
    """和 tools/verify_0526.py 同款参考实现（letterbox / 解码 / NMS）"""
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"读不到图片: {path}")
    h, w = img.shape[:2]
    scale = min(640 / h, 640 / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    x = canvas[:, :, ::-1].astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    inp = sess.get_inputs()[0]
    if "float16" in inp.type:
        x = x.astype(np.float16)
    out = sess.run(None, {inp.name: x})[0][0].astype(np.float32)

    conf = 1.0 / (1.0 + np.exp(-out[:, 8]))
    cand = np.where(conf > conf_th)[0]
    boxes, scores, dets = [], [], []
    for i in cand:
        row = out[i]
        c, n = int(np.argmax(row[9:13])), int(np.argmax(row[13:22]))
        pts = [(row[0] / scale, row[1] / scale), (row[6] / scale, row[7] / scale),
               (row[4] / scale, row[5] / scale), (row[2] / scale, row[3] / scale)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        boxes.append([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)])
        scores.append(float(conf[i]))
        dets.append({"color": PY_COLORS[c], "num": PY_NUMS[n], "conf": float(conf[i]),
                     "pts": pts, "center": (sum(xs) / 4, sum(ys) / 4)})
    if boxes:
        keep = np.array(cv2.dnn.NMSBoxes(boxes, scores, conf_th, NMS_TH)).flatten().tolist()
        dets = [dets[int(k)] for k in keep]
    return dets


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cpp_file, images = sys.argv[1], sys.argv[2:]
    model = os.environ.get("MODEL", "models/0526.onnx")
    cpp_all = parse_cpp(cpp_file)
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    def center_of(d):
        return (sum(q[0] for q in d["pts"]) / 4, sum(q[1] for q in d["pts"]) / 4)

    n_img = n_pairs = n_edge = 0
    bad, warn = [], []
    for path in images:
        base = os.path.basename(path)
        n_img += 1
        py = reference(sess, path)
        cc = cpp_all.get(base)
        if cc is None:
            bad.append(f"{base}: C++ 输出里没有这张图（yolo_test 跑过它吗？）")
            continue
        print(f"\n{base}   python {len(py)} 个 / C++ {len(cc)} 个")
        used = set()
        for p in py:
            # 按中心就近配对
            best, bd = None, 1e9
            for j, c in enumerate(cc):
                if j in used or "pts" not in c:
                    continue
                if "pts" not in p:
                    continue
                pc = (sum(q[0] for q in p["pts"]) / 4, sum(q[1] for q in p["pts"]) / 4)
                cxc = (sum(q[0] for q in c["pts"]) / 4, sum(q[1] for q in c["pts"]) / 4)
                d = ((pc[0] - cxc[0]) ** 2 + (pc[1] - cxc[1]) ** 2) ** 0.5
                if d < bd:
                    best, bd = j, d
            if best is None or bd > 40:
                desc = f"python {p['color']}/{p['num']} conf={p['conf']:.3f}"
                if p["conf"] <= CONF_TH + CONF_TOL:
                    n_edge += 1
                    warn.append(
                        f"{base}: {desc} 仅高出阈值 {p['conf'] - CONF_TH:.3f}，C++ 侧没有 → 疑似阈值边缘")
                    print(f"  [~~] {desc} → C++ 侧无对应（疑似阈值边缘；需确认可调低 C++ 侧阈值复跑）")
                else:
                    bad.append(
                        f"{base}: python 检出 {p['color']}/{p['num']} 在 C++ 侧找不到对应（最近 {bd:.0f}px）")
                    print(f"  [!!] {desc} → C++ 侧无对应")
                continue
            used.add(best)
            c = cc[best]
            n_pairs += 1
            # 颜色
            pexpect_color = {"gray": "extinguish"}.get(p["color"], p["color"])
            cname, ctype = NUM_MAP[p["num"]]
            ok_color = pexpect_color == c["color"]
            ok_num = (c["name"] == cname) and (ctype is None or c["type"] == ctype)
            ok_conf = abs(p["conf"] - c["conf"]) <= CONF_TOL
            pt_d = max(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(p["pts"], c["pts"]))
            ok_pt = pt_d <= PT_TOL
            tag = "OK " if (ok_color and ok_num and ok_conf and ok_pt) else "!! "
            if tag == "!! ":
                bad.append(
                    f"{base}: python {p['color']}/{p['num']} vs C++ {c['color']}/{c['name']}/{c['type']} "
                    f"conf {p['conf']:.3f}/{c['conf']:.3f} 角点差 {pt_d:.1f}px")
            print(f"  [{tag}] python {p['color']:5s}/{p['num']:2s} {p['conf']:.3f}   "
                  f"C++ {c['color']:10s}/{c['name']:8s}/{c['type']:5s} {c['conf']:.3f}   "
                  f"Δconf={abs(p['conf']-c['conf']):.3f}  角点最大差={pt_d:.1f}px   配对距={bd:.1f}px")
            print(f"        py 角点 {[(round(a,1),round(b,1)) for a,b in p['pts']]}")
            print(f"        cc 角点 {[(round(a,1),round(b,1)) for a,b in c['pts']]}")
        extra = [c for j, c in enumerate(cc) if j not in used]
        if extra:
            # 低阈值复跑 python：同位置有候选、且 conf 只差 ≤ 0.03 → 阈值边缘（浮点路径一取一舍），不算失败
            loose = reference(sess, path, conf_th=CONF_TH - CONF_TOL)
            for c in extra:
                cxc = center_of(c)
                hit = None  # (conf, 距中心 px)
                for p in loose:
                    if "pts" not in p:
                        continue
                    pc = center_of(p)
                    d = ((pc[0] - cxc[0]) ** 2 + (pc[1] - cxc[1]) ** 2) ** 0.5
                    if d <= 40 and (hit is None or p["conf"] > hit[0]):
                        hit = (p["conf"], d)
                desc = f"C++ {c['color']}/{c['name']}/{c['type']} conf={c['conf']:.3f}"
                if hit and abs(hit[0] - c["conf"]) <= CONF_TOL:
                    n_edge += 1
                    warn.append(f"{base}: {desc}，python 低阈值复跑同位置 conf={hit[0]:.3f} → 阈值边缘（非解码错）")
                    print(f"  [~~] {desc} → python 低阈值复跑同位置 conf={hit[0]:.3f}（阈值边缘，非解码错）")
                else:
                    got = f"低阈值复跑最近 {hit[0]:.3f}（距 {hit[1]:.0f}px）" if hit else "低阈值复跑也没有"
                    bad.append(f"{base}: {desc} 在 python 侧没有（{got}）")
                    print(f"  [!!] {desc} → python 侧无对应（{got}）")

    print(f"\n==== 汇总：{n_img} 张图，配对 {n_pairs} 个检出，阈值边缘 {n_edge} 例 ====")
    for w in warn:
        print("  [~~]", w)
    if bad:
        print("不通过：")
        for b in bad:
            print("  -", b)
        sys.exit(1)
    print("全部通过 ✓（阈值边缘属预期浮点差）")


if __name__ == "__main__":
    main()
