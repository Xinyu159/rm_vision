#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大恒相机参数一次性摸底（只读）：型号/分辨率/ROI/像素格式/帧率/曝光/增益/带宽。

用 Galaxy SDK 自带 Python 绑定（gxipy，ctypes 版，免编译）。只读不写。
运行: PYTHONPATH=<sdk>/Galaxy_Linux_Python_*/api python3 gx_dump.py
"""
import sys


def main():
    import gxipy as gx
    from gxipy.Feature import Feature, BufferFeature

    dm = gx.DeviceManager()
    num, infos = dm.update_device_list()
    print("== 枚举到相机数量: %d ==" % num)
    for info in infos:
        print("   ", info)
    if num == 0:
        print("!! 没发现相机（检查 USB 连接）")
        return 1

    cam = None
    try:
        cam = dm.open_device_by_index(1)
    except Exception as e:
        print("按序号打开失败: %r，尝试按 SN 打开" % e)
        sn = infos[0].get("sn") or infos[0].get("SN")
        if sn:
            cam = dm.open_device_by_sn(str(sn).strip())
    if cam is None:
        print("!! 打不开相机")
        return 1

    try:
        # ---- 收集所有 Feature 属性 ----
        names = []
        for n in dir(cam):
            if n.startswith("_"):
                continue
            try:
                f = getattr(cam, n)
            except Exception:
                continue
            if isinstance(f, Feature) and not isinstance(f, BufferFeature):
                names.append(n)
        print("\n== Feature 属性数: %d ==" % len(names))
        if not names:
            print("目录前 60 项:", [n for n in dir(cam) if not n.startswith("_")][:60])

        print("\n== 全部可实现特征（只读遍历）==")
        for n in sorted(names):
            f = getattr(cam, n)
            try:
                if not f.is_implemented():
                    continue
            except Exception:
                continue
            line = "%-36s" % n
            try:
                val = f.get()
                if isinstance(val, bytes):
                    val = val.decode("utf-8", "replace")
                line += " = %s" % (val,)
            except Exception as e:
                line += " <get失败 %s>" % type(e).__name__
            try:
                rng = f.get_range()
                if rng:
                    line += "    range=%s" % (rng,)
            except Exception:
                pass
            print(line)

        # ---- 关键摘要 ----
        print("\n== 关键摘要 ==")
        key = ["DeviceVendorName", "DeviceModelName", "DeviceSerialNumber",
               "DeviceFirmwareVersion", "DeviceVersion", "DeviceUserID",
               "WidthMax", "HeightMax", "Width", "Height", "OffsetX", "OffsetY",
               "PixelFormat", "PixelSize",
               "AcquisitionFrameRateEnable", "AcquisitionFrameRate",
               "AcquisitionMode", "TriggerMode",
               "ExposureTime", "ExposureAuto", "Gain", "GainAuto",
               "BalanceWhiteAuto", "BalanceRatio",
               "DeviceLinkThroughputLimit", "DeviceLinkThroughputLimitMode",
               "PayloadSize", "AcquisitionSpeedLevel"]
        for n in key:
            try:
                f = getattr(cam, n)
            except Exception:
                print("%-34s <无此特征>" % n)
                continue
            seg = "%-34s" % n
            try:
                v = f.get()
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace")
                seg += " = %s" % (v,)
            except Exception as e:
                seg += " <get失败 %s>" % type(e).__name__
            try:
                rng = f.get_range()
                if rng:
                    seg += "    range=%s" % (rng,)
            except Exception:
                pass
            print(seg)
    finally:
        cam.close_device()
        print("\n(相机已关闭)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
