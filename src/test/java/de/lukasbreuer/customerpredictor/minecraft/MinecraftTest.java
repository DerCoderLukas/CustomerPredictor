package de.lukasbreuer.customerpredictor.minecraft;

import java.util.GregorianCalendar;
import java.util.List;

import com.google.common.collect.Lists;

import de.lukasbreuer.customerpredictor.NeuralNetwork;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

final class MinecraftTest {
  private NeuralNetwork<MiniGameMatch> neuralNetwork;

  // IN PRACTICE, THE ACCESS TO THE DATA IS OF COURSE HANDLED VIA A DATABASE
  private static final List<MiniGameMatch> MATCH_HISTORY = Lists.newArrayList(
    MiniGameMatch.create(MiniGame.BEDWARS, new GregorianCalendar(2021, 0, 0)),
    MiniGameMatch.create(MiniGame.SKYWARS, new GregorianCalendar(2021, 0, 1)),
    MiniGameMatch.create(MiniGame.BEDWARS, new GregorianCalendar(2021, 0, 2)),
    MiniGameMatch.create(MiniGame.SKYWARS, new GregorianCalendar(2021, 0, 3)));

  @BeforeEach
  void initializeNeuralNetwork() throws Exception {
    neuralNetwork = NeuralNetwork.create(MATCH_HISTORY);
    neuralNetwork.train(1000);
  }

  @Test
  void predictNextMatch() throws Exception {
    Assertions.assertEquals(neuralNetwork.predict(MATCH_HISTORY,
      MATCH_HISTORY.get(3)).findMiniGame(), MiniGame.BEDWARS);
  }
}
