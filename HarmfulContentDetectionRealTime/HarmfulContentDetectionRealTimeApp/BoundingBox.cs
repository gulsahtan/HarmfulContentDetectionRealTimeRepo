using System.Drawing;

namespace HarmfulContentDetectionRealTimeApp
{
    public class BoundingBox
    {
        public BoundingBox()
        {
        }

        public BoundingBox(BoundingBoxDimensions dimensions, string label, float confidence, Color boxColor)
        {
            Dimensions = dimensions;
            Label = label;
            Confidence = confidence;
            BoxColor = boxColor;
        }


        public BoundingBoxDimensions Dimensions { get; set; }

        public string Label { get; set; }


        public float Confidence { get; set; }


        public Color BoxColor { get; set; }


        public string Description
        {
            get
            {
                return $"{Label} ({Confidence * 100:0}%)";
            }
        }

        public RectangleF Rect
        {
            get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
        }
    }
}