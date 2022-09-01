using System.Drawing;

namespace HarmfulContentDetectionRealTimeApp.Utils
{
   public static class ColorArray
   {

      private static readonly Color[] classColors = new Color[]
      {
         Color.Khaki,
         Color.Fuchsia,
         Color.Silver,
         Color.RoyalBlue,
         Color.Green,
         Color.DarkOrange,
         Color.Purple,
         Color.Gold,
         Color.Red,

         Color.Aquamarine,
         Color.Lime,
         Color.AliceBlue,
         Color.Sienna,
         Color.Orchid,
         Color.Tan,
         Color.LightPink,
         Color.Yellow,
         Color.HotPink,
         Color.OliveDrab,
         Color.SandyBrown,
         Color.DarkTurquoise
      };
    
      public static Color GetColor(int index)
      {
         return index < classColors.Length ? classColors[index] : classColors[index % classColors.Length];
      }
   }
}
