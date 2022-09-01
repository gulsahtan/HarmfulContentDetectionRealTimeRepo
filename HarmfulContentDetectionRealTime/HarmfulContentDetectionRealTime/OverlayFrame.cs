using HarmfulContentDetectionRealTime;
using HarmfulContentDetectionRealTimeApp;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace HarmfulContentDetectionRealTime
{
    internal class OverlayFrame
    {
        public static void DrawOverlays(ICanvasHandler canvasHandler, List<BoundingBox> boundingBoxes, double originalHeight, double originalWidth)
        {
            canvasHandler.Clear();

            foreach (BoundingBox box in boundingBoxes)
            {
                double x = Math.Max(box.Dimensions.X, 0);
                double y = Math.Max(box.Dimensions.Y, 0);
                double width = Math.Min(originalWidth - x, box.Dimensions.Width);
                double height = Math.Min(originalHeight - y, box.Dimensions.Height);

                x = originalWidth * x / 640;
                y = originalHeight * y / 640;
                width = originalWidth * width / 640;
                height = originalHeight * height / 640;

                var boxColor = box.BoxColor.ToMediaColor();

                var objBox = GetObjBox(x, y, width, height, boxColor);

                var objDescription = new TextBlock
                {
                    Margin = new Thickness(x + 4, y + 4, 0, 0),
                    Text = box.Description,
                    FontWeight = FontWeights.Bold,
                    Width = 126,
                    Height = 21,
                    TextAlignment = TextAlignment.Center
                };

                var objDescriptionBackground = new Rectangle
                {
                    Width = 134,
                    Height = 29,
                    Fill = new SolidColorBrush(boxColor),
                    Margin = new Thickness(x, y, 0, 0)
                };

                canvasHandler.AddToCanvas(objDescriptionBackground);
                canvasHandler.AddToCanvas(objDescription);
                canvasHandler.AddToCanvas(objBox);
            }
        }

        private static Rectangle GetObjBox(double x, double y, double width, double height, System.Windows.Media.Color boxColor)
        {
            if(height > 0 && width > 0)
            {           
            return new Rectangle
            {
                Width = width,
                Height = height,
                Fill = new SolidColorBrush(Colors.Transparent),
                Stroke = new SolidColorBrush(boxColor),
                StrokeThickness = 2.0,
                Margin = new Thickness(x, y, 0, 0)
            };
            }
            else
            {
                return new Rectangle
                {
                    Width = 0,
                    Height = 0,
                    Fill = new SolidColorBrush(Colors.Transparent),
                    Stroke = new SolidColorBrush(boxColor),
                    StrokeThickness = 2.0,
                    Margin = new Thickness(x, y, 0, 0)
                };
            }
        }
    }
}
