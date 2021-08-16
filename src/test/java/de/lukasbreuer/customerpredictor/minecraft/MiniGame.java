package de.lukasbreuer.customerpredictor.minecraft;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

@Getter
@Accessors(fluent = true)
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
enum MiniGame {
  // IN THAT CASE I USE ONE HOT ENCODED VECTORS BUT WITH MORE COMPLEX USE
  // STRUCTURES IT WOULD BE BETTER TO CLUSTER THE USES VIA OWN VECTOR
  // ALGORITHMS
  BEDWARS(new float[] {1, 0}),
  SKYWARS(new float[] {0, 1});

  private final float[] vector;
}
