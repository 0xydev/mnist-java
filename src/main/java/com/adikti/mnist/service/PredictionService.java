package com.adikti.mnist.service;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.PairList;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.IntStream;

@Service
public class PredictionService {

    private static final Logger log = LoggerFactory.getLogger(PredictionService.class);
    private static boolean hasTrainedModel() {
        try (var files = Files.list(TrainingService.MODEL_DIR)) {
            return files.anyMatch(p -> p.getFileName().toString().matches(
                    TrainingService.MODEL_NAME + "-\\d+\\.params"));
        } catch (IOException e) {
            return false;
        }
    }

    private Model model;
    private Predictor<Image, Classifications> predictor;

    @PostConstruct
    public void init() {
        if (hasTrainedModel()) {
            try {
                loadModel();
                log.info("Model loaded successfully");
            } catch (Exception e) {
                log.warn("Failed to load model: {}", e.getMessage());
            }
        } else {
            log.info("No trained model found. Use POST /api/train first.");
        }
    }

    @PreDestroy
    public void destroy() {
        if (predictor != null) predictor.close();
        if (model != null) model.close();
    }

    public void loadModel() throws IOException, MalformedModelException {
        if (predictor != null) predictor.close();
        if (model != null) model.close();

        model = Model.newInstance(TrainingService.MODEL_NAME);
        model.setBlock(TrainingService.createCnnBlock());
        model.load(TrainingService.MODEL_DIR, TrainingService.MODEL_NAME);
        predictor = model.newPredictor(new MnistTranslator());
    }

    public Classifications predict(InputStream imageStream) throws Exception {
        if (predictor == null) {
            throw new IllegalStateException("Model not loaded. Use POST /api/train first.");
        }
        return predictor.predict(ImageFactory.getInstance().fromInputStream(imageStream));
    }

    public String getProcessedImageBase64(InputStream imageStream) throws Exception {
        Image image = ImageFactory.getInstance().fromInputStream(imageStream);

        try (NDManager manager = NDManager.newBaseManager()) {
            float[] pixels = centerDigit(image, manager);

            BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int val = Math.min(255, Math.max(0, (int) (pixels[y * 28 + x] * 255)));
                    bi.setRGB(x, y, (val << 16) | (val << 8) | val);
                }
            }

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(bi, "png", baos);
            return Base64.getEncoder().encodeToString(baos.toByteArray());
        }
    }

    public PairList<String, Parameter> getModelParameters() {
        if (model == null) {
            throw new IllegalStateException("Model not loaded. Use POST /api/train first.");
        }
        return model.getBlock().getParameters();
    }

    public List<Map<String, Object>> exportWeights() {
        if (model == null) {
            throw new IllegalStateException("Model not loaded. Use POST /api/train first.");
        }

        List<Map<String, Object>> params = new ArrayList<>();
        PairList<String, Parameter> parameters = model.getBlock().getParameters();

        for (var pair : parameters) {
            NDArray array = pair.getValue().getArray();
            long[] shape = array.getShape().getShape();
            float[] data = array.toFloatArray();

            ByteBuffer buffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : data) buffer.putFloat(v);

            params.add(Map.of(
                    "name", pair.getKey(),
                    "shape", shape,
                    "data", Base64.getEncoder().encodeToString(buffer.array())
            ));
        }

        return params;
    }

    static float[] centerDigit(Image image, NDManager manager) {
        NDArray array = image.toNDArray(manager, Image.Flag.GRAYSCALE);
        int h = (int) array.getShape().get(0);
        int w = (int) array.getShape().get(1);
        float[] pixels = array.toType(DataType.FLOAT32, false).toFloatArray();

        int minX = w, minY = h, maxX = 0, maxY = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (pixels[y * w + x] > 20f) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        if (minX > maxX) return new float[28 * 28];

        int cropW = maxX - minX + 1;
        int cropH = maxY - minY + 1;
        float[] cropped = new float[cropH * cropW];
        for (int y = 0; y < cropH; y++) {
            for (int x = 0; x < cropW; x++) {
                cropped[y * cropW + x] = pixels[(minY + y) * w + (minX + x)];
            }
        }

        int targetSize = 20;
        int newW, newH;
        if (cropW > cropH) {
            newW = targetSize;
            newH = Math.max(1, (int) Math.round((double) cropH / cropW * targetSize));
        } else {
            newH = targetSize;
            newW = Math.max(1, (int) Math.round((double) cropW / cropH * targetSize));
        }

        NDArray cropArray = manager.create(cropped, new Shape(cropH, cropW)).expandDims(2);
        cropArray = NDImageUtils.resize(cropArray, newW, newH).squeeze(2);
        float[] resized = cropArray.toType(DataType.FLOAT32, false).toFloatArray();

        float[] result = new float[28 * 28];
        int offsetX = (28 - newW) / 2;
        int offsetY = (28 - newH) / 2;

        for (int y = 0; y < newH; y++) {
            for (int x = 0; x < newW; x++) {
                result[(offsetY + y) * 28 + (offsetX + x)] = resized[y * newW + x] / 255f;
            }
        }

        return result;
    }

    private static class MnistTranslator implements Translator<Image, Classifications> {

        private static final List<String> CLASSES =
                IntStream.range(0, 10).mapToObj(String::valueOf).toList();

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            float[] centered = centerDigit(input, ctx.getNDManager());
            NDArray array = ctx.getNDManager().create(centered, new Shape(1, 28, 28));
            return new NDList(array.sub(TrainingService.MNIST_MEAN).div(TrainingService.MNIST_STD));
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            return new Classifications(CLASSES, list.singletonOrThrow().softmax(0));
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
}
