package de.lukasbreuer.customerpredictor;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

@Accessors(fluent = true)
@RequiredArgsConstructor(access = AccessLevel.PROTECTED)
public abstract class Use {
  @Getter
  private final float[] vector;

  public abstract float[] attributeVector();

  public float[] fullVector() {
    var output = new float[vector.length + attributeVector().length];
    System.arraycopy(vector, 0, output, 0, vector.length);
    System.arraycopy(attributeVector(), 0, output, vector.length, attributeVector().length);
    return output;
  }
}
