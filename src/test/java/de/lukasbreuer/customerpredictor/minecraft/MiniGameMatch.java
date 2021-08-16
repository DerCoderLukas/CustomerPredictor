package de.lukasbreuer.customerpredictor.minecraft;

import java.util.Arrays;
import java.util.Calendar;

import de.lukasbreuer.customerpredictor.Use;

final class MiniGameMatch extends Use {
  public static MiniGameMatch create(MiniGame miniGame, Calendar date) {
    return create(miniGame.vector(), date);
  }

  public static MiniGameMatch create(float[] vector, Calendar date) {
    return new MiniGameMatch(vector, date);
  }

  // FOR THE APPLICATION IN A PRODUCTIVE ENVIRONMENT IT IS RECOMMENDED TO
  // COLLECT AS MANY ATTRIBUTES ABOUT THE USE AS POSSIBLE TO GET THE BEST
  // RESULTS
  private final Calendar date;

  private MiniGameMatch(float[] vector, Calendar date) {
    super(vector);
    this.date = date;
  }

  @Override
  public float[] attributeVector() {
    return new float[] {relativizeValue(date.get(Calendar.YEAR), 2100),
      relativizeValue(date.get(Calendar.MONTH), 12),
      relativizeValue(date.get(Calendar.DAY_OF_MONTH), 31),
      relativizeValue(date.get(Calendar.DAY_OF_WEEK), 7)};
  }

  private float relativizeValue(int value, int range) {
    return (float) value / (float) range;
  }

  public MiniGame findMiniGame() {
    return Arrays.stream(MiniGame.values())
      .filter(miniGame -> Arrays.equals(miniGame.vector(), vector()))
      .findFirst().get();
  }
}
