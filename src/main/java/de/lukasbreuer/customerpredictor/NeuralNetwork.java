package de.lukasbreuer.customerpredictor;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.common.primitives.Floats;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor(staticName = "create")
public final class NeuralNetwork<T extends Use> {
  public static <T extends Use> NeuralNetwork<T> create(List<T> previousUses) {
      var firstUse = previousUses.get(0);
      Preconditions.checkNotNull(firstUse);
      var vectorSize = firstUse.vector().length;
      return create(previousUses, "customer_predictor", firstUse.fullVector().length,
        new int[] {100, 100}, vectorSize);
  }

  public static <T extends Use> NeuralNetwork<T> create(
    List<T> previousUses, String modelName, int inputNeurons,
    int[] hiddenNeurons, int outputNeurons
  ) {
    var model = Model.newInstance(modelName);
    model.setBlock(createNetwork(inputNeurons, hiddenNeurons, outputNeurons));
    return create(UseDataset.create(previousUses),
      inputNeurons, hiddenNeurons, outputNeurons, model);
  }

  private final UseDataset<T> dataset;
  private final int inputNeurons;
  private final int[] hiddenNeurons;
  private final int outputNeurons;
  private final Model model;

  public void train(int epochs) throws Exception {
    var trainer = model.newTrainer(configureTrainer());
    trainer.initialize(new Shape(inputNeurons));
    EasyTrain.fit(trainer, epochs, dataset, null);
  }

  private TrainingConfig configureTrainer() {
    return new DefaultTrainingConfig(
      Loss.softmaxCrossEntropyLoss("SoftmaxCrossEntropyLoss", 1, -1, false, true))
      .addEvaluator(new Accuracy());
  }

  public T predict(List<T> previousUses, T previousUse) throws Exception {
    var predictor = model.newPredictor(new PhrasesTranslator());
    return findMostLikelyUse(previousUses, Floats.toArray(Arrays.stream(
      predictor.predict(NDManager.newBaseManager().create(previousUse.fullVector())))
      .collect(Collectors.toList())));
  }

  private T findMostLikelyUse(List<T> previousUses, float[] useVector) {
    var useDistances = Maps.<T, Float>newHashMap();
    for (var use : previousUses) {
      var accumulatedDistance = 0f;
      for (var i = 0; i < use.vector().length; i++) {
        accumulatedDistance += Math.abs(useVector[i] - use.vector()[i]);
      }
      useDistances.put(use, accumulatedDistance);
    }
    return useDistances.entrySet().stream()
      .sorted(Map.Entry.comparingByValue())
      .map(Map.Entry::getKey).findFirst().get();
  }

  private static SequentialBlock createNetwork(
    int inputNeurons, int[] hiddenNeurons, int outputNeurons
  ) {
    var block = new SequentialBlock();
    block.add(Blocks.batchFlattenBlock(inputNeurons));
    for (var hiddenSize : hiddenNeurons) {
      block.add(Linear.builder().setUnits(hiddenSize).build());
    }
    block.add(Linear.builder().setUnits(outputNeurons).build());
    return block;
  }

  private final class PhrasesTranslator implements Translator<NDArray, Float[]> {
    @Override
    public NDList processInput(TranslatorContext ctx, NDArray input) {
      return new NDList(input);
    }

    @Override
    public Float[] processOutput(TranslatorContext ctx, NDList list) {
      return Arrays.stream(list.get(0).toArray())
        .map(Number::floatValue)
        .map(value -> (float) (1 / (1 + Math.exp(-value))))
        .toArray(Float[]::new);
    }

    @Override
    public Batchifier getBatchifier() {
      return Batchifier.STACK;
    }
  }
}
