public sealed class PointElement: ElementBase{
  private int x_;
  private int y_;

  public override void MoveTo(int x, int y){
    x_ = x;
    y_ = y;
  }

  public override void Draw(IDrawEngine engine){
    engine.DrawPoint(x_, y_);
  }
}

public sealed class LineElement: ElementBase{
  private int lastX_;
  private int lastY_;
  private int x_;
  private int y_;

  public override void MoveTo(int x, int y){
    lastX_ = x_;
    lastY_ = y_;

    x_ = x;
    y_ = y;
  }

  public override void Draw(IDrawEngine engine){
    engine.DrawLine(lastX_, lastY_, x_, y_);
  }
}

public abstract class ElementBase{
  protected ElementBase(){}

  public abstract void MoveTo(int x, int y);
  public abstract void Draw(IDrawEngine engine)
}

private static void DrawPolyline(IDrawEngine engine, ElementBase element,
    IEnumerable<Tuple<int, int>> points){
  foreach(var entry in points){
    element.MoveTo(entry.Item1, entry.Item2);
    element.Draw(engine)
  }
}

var points = new[]{Tuple.Create(1, 1), Tuple.Create(5, 4), Tuple.Create(7, 3)}

DrawPolyline(engine, new PointElement(), points);
DrawPolyline(engine, new LineElement(), points);