package com.n24q02m.yolodetectncnn;

import android.content.res.AssetManager;
import android.view.Surface;

public class Yolov8Ncnn {
    public native boolean loadModel(AssetManager mgr, int modelid);

    public native boolean openCamera(int facing);

    public native boolean closeCamera();

    public native boolean setOutputWindow(Surface surface);

    static {
        System.loadLibrary("yolov8ncnn");
    }
}