package com.adikti.mnist.controller;

import ai.djl.modality.Classifications;
import com.adikti.mnist.service.PredictionService;
import com.adikti.mnist.service.TrainingService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class MnistController {

    private final TrainingService trainingService;
    private final PredictionService predictionService;

    public MnistController(TrainingService trainingService, PredictionService predictionService) {
        this.trainingService = trainingService;
        this.predictionService = predictionService;
    }

    @PostMapping("/train")
    public ResponseEntity<Map<String, String>> train() {
        try {
            trainingService.train();
            predictionService.loadModel();
            return ResponseEntity.ok(Map.of("status", "success", "message", "Model trained and loaded."));
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body(Map.of("status", "error", "message", e.getMessage()));
        }
    }

    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> predict(@RequestParam("image") MultipartFile image) {
        try {
            byte[] imageBytes = image.getBytes();
            Classifications result = predictionService.predict(new ByteArrayInputStream(imageBytes));
            String debugImage = predictionService.getProcessedImageBase64(new ByteArrayInputStream(imageBytes));
            Classifications.Classification best = result.best();

            return ResponseEntity.ok(Map.of(
                    "prediction", best.getClassName(),
                    "confidence", String.format("%.2f%%", best.getProbability() * 100),
                    "processedImage", "data:image/png;base64," + debugImage,
                    "top5", result.topK(10).stream()
                            .map(c -> Map.of(
                                    "digit", c.getClassName(),
                                    "probability", String.format("%.4f", c.getProbability())))
                            .toList()
            ));
        } catch (IllegalStateException e) {
            return ResponseEntity.badRequest().body(Map.of("status", "error", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body(Map.of("status", "error", "message", e.getMessage()));
        }
    }

    @GetMapping("/export")
    public ResponseEntity<?> exportWeights() {
        try {
            List<Map<String, Object>> weights = predictionService.exportWeights();
            return ResponseEntity.ok(Map.of("parameters", weights));
        } catch (IllegalStateException e) {
            return ResponseEntity.badRequest().body(Map.of("status", "error", "message", e.getMessage()));
        }
    }
}
