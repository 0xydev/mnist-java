package com.adikti.mnist.service;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.translate.Pipeline;
import com.adikti.mnist.model.RandomAugmentation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@Service
public class TrainingService {

    private static final Logger log = LoggerFactory.getLogger(TrainingService.class);

    public static final float MNIST_MEAN = 0.1307f;
    public static final float MNIST_STD = 0.3081f;
    public static final String MODEL_NAME = "mnist";
    public static final Path MODEL_DIR = Paths.get("models");

    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;

    public static SequentialBlock createCnnBlock() {
        SequentialBlock block = new SequentialBlock();

        block.add(Conv2d.builder().setFilters(32).setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        block.add(Conv2d.builder().setFilters(64).setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        block.add(Blocks.batchFlattenBlock());
        block.add(Linear.builder().setUnits(128).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(10).build());

        return block;
    }

    public void train() throws Exception {
        log.info("MNIST CNN training started");

        RandomAccessDataset trainSet = prepareDataset(Dataset.Usage.TRAIN);
        RandomAccessDataset testSet = prepareDataset(Dataset.Usage.TEST);

        try (Model model = Model.newInstance(MODEL_NAME)) {
            model.setBlock(createCnnBlock());

            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(new Shape(BATCH_SIZE, 1, 28, 28));
                EasyTrain.fit(trainer, EPOCHS, trainSet, testSet);

                log.info("Training completed — Train: {}, Test: {}",
                        trainer.getTrainingResult().getTrainEvaluation("Accuracy"),
                        trainer.getTrainingResult().getValidateEvaluation("Accuracy"));
            }

            Files.createDirectories(MODEL_DIR);
            model.save(MODEL_DIR, MODEL_NAME);
            log.info("Model saved to {}", MODEL_DIR.toAbsolutePath());
        }
    }

    private RandomAccessDataset prepareDataset(Dataset.Usage usage) throws Exception {
        Pipeline pipeline = new Pipeline();
        pipeline.add(new ToTensor());

        if (usage == Dataset.Usage.TRAIN) {
            pipeline.add(new RandomAugmentation(3, 15));
        }

        pipeline.add(array -> array.sub(MNIST_MEAN).div(MNIST_STD));

        Mnist dataset = Mnist.builder()
                .optUsage(usage)
                .setSampling(BATCH_SIZE, true)
                .optPipeline(pipeline)
                .build();
        dataset.prepare();
        return dataset;
    }
}
