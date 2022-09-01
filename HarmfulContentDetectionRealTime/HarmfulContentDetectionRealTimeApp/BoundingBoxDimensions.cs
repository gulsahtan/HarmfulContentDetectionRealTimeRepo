namespace HarmfulContentDetectionRealTimeApp
{
   public class BoundingBoxDimensions
   {
      public BoundingBoxDimensions()
      {
      }

      public BoundingBoxDimensions(float x, float y, float height, float width)
      {
         X = x;
         Y = y;
         Height = height;
         Width = width;
      }

      public float X { get; set; }

      public float Y { get; set; }   

      public float Height { get; set; }

      public float Width { get; set; }
   }
}