﻿using System.Drawing;

namespace HarmfulContentDetectionRealTime
{

    public class Prediction
    {
        public Label Label { get; set; }
        public RectangleF Rectangle { get; set; }
        public float Score { get; set; }

        public Prediction() { }

        public Prediction(Label label, float confidence) : this(label)
        {
            Score = confidence;
        }

        public Prediction(Label label)
        {
            Label = label;
        }
    }
}
