package com.adikti.mnist.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Transform;

import java.util.Arrays;
import java.util.Random;

public class RandomAugmentation implements Transform {

    private final int maxShift;
    private final double maxAngleRad;
    private final Random random = new Random();

    public RandomAugmentation(int maxShift, double maxAngleDegrees) {
        this.maxShift = maxShift;
        this.maxAngleRad = maxAngleDegrees * Math.PI / 180.0;
    }

    @Override
    public NDArray transform(NDArray array) {
        int dims = array.getShape().dimension();

        if (dims == 4) {
            int n = (int) array.getShape().get(0);
            NDManager manager = array.getManager();
            NDArray[] results = new NDArray[n];
            for (int i = 0; i < n; i++) {
                results[i] = augmentSingle(array.get(i), manager);
            }
            return NDArrays.stack(new NDList(results));
        }

        if (dims == 3) {
            return augmentSingle(array, array.getManager());
        }

        return array;
    }

    private NDArray augmentSingle(NDArray sample, NDManager manager) {
        Shape shape = sample.getShape();
        int c = (int) shape.get(0);
        int h = (int) shape.get(1);
        int w = (int) shape.get(2);
        int size = c * h * w;

        int dx = random.nextInt(2 * maxShift + 1) - maxShift;
        int dy = random.nextInt(2 * maxShift + 1) - maxShift;
        double angle = (random.nextDouble() * 2 - 1) * maxAngleRad;

        if (dx == 0 && dy == 0 && Math.abs(angle) < 0.001) {
            return sample;
        }

        float[] rawData = sample.toFloatArray();
        float[] src = (rawData.length == size) ? rawData : Arrays.copyOf(rawData, size);
        float[] dst = new float[size];

        double cosA = Math.cos(angle);
        double sinA = Math.sin(angle);
        double cx = w / 2.0;
        double cy = h / 2.0;

        for (int ch = 0; ch < c; ch++) {
            int offset = ch * h * w;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    double rx = x - cx;
                    double ry = y - cy;
                    double srcX = cosA * rx + sinA * ry + cx - dx;
                    double srcY = -sinA * rx + cosA * ry + cy - dy;
                    dst[offset + y * w + x] = bilinearSample(src, offset, h, w, srcX, srcY);
                }
            }
        }

        return manager.create(dst, shape);
    }

    private float bilinearSample(float[] data, int offset, int h, int w, double x, double y) {
        int x0 = (int) Math.floor(x);
        int y0 = (int) Math.floor(y);
        float fx = (float) (x - x0);
        float fy = (float) (y - y0);

        return (1 - fx) * (1 - fy) * pixelAt(data, offset, h, w, x0, y0)
                + fx * (1 - fy) * pixelAt(data, offset, h, w, x0 + 1, y0)
                + (1 - fx) * fy * pixelAt(data, offset, h, w, x0, y0 + 1)
                + fx * fy * pixelAt(data, offset, h, w, x0 + 1, y0 + 1);
    }

    private float pixelAt(float[] data, int offset, int h, int w, int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) return 0f;
        return data[offset + y * w + x];
    }
}
