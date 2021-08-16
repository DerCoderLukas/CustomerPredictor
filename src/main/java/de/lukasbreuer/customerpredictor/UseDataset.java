package de.lukasbreuer.customerpredictor;

import java.util.List;

import com.google.common.collect.Lists;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;

public final class UseDataset<T extends Use> extends RandomAccessDataset {
  public static <T extends Use> UseDataset<T> create(List<T> previousUses) {
    return new UseDataset<T>(new Builder().setSampling(32, true),
      dataOfPreviousUses(previousUses));
  }

  private final List<NextUseContext<T>> data;

  private UseDataset(
    BaseBuilder<?> builder, List<NextUseContext<T>> data
  ) {
    super(builder);
    this.data = data;
  }

  @Override
  public Record get(NDManager manager, long index) {
    var datum = new NDList();
    var label = new NDList();
    var nextUseContext = data.get((int) index);
    datum.add(manager.create(nextUseContext.previousUse().fullVector()));
    label.add(manager.create(nextUseContext.nextUse().vector()));
    datum.attach(manager);
    label.attach(manager);
    return new Record(datum, label);
  }

  @Override
  protected long availableSize() {
    return data.size();
  }

  @Override
  public void prepare(Progress progress) {}

  private static final class Builder extends BaseBuilder<Builder> {
    @Override
    protected Builder self() {
      return this;
    }
  }

  public record NextUseContext<T extends Use>(T previousUse, T nextUse) {}

  private static <T extends Use> List<NextUseContext<T>> dataOfPreviousUses(
    List<T> previousUses
  ) {
    var data = Lists.<NextUseContext<T>>newArrayList();
    for (var i = 1; i < previousUses.size(); i++) {
      var use = previousUses.get(i);
      data.add(new NextUseContext<T>(previousUses.get(i - 1), use));
    }
    return data;
  }
}
