using System.Drawing;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;

namespace TrafficLight
{
    public class Car
    {
        [JsonPropertyName("x")]
        public double X { get; set; }
        [JsonPropertyName("y")]
        public double Y { get; set; }
        [JsonPropertyName("ratio")]
        public double Ratio { get; set; }
        [JsonPropertyName("lane")]
        public int Lane { get; set; }
        public Color Color { get; set; }
        public Car(double x, double y, double ratio, int lane)
        {
            Color = Color.FromArgb(new Random().Next(256), new Random().Next(256), new Random().Next(256));
            X = x;
            Y = y;
            Ratio = ratio;
            Lane = lane;
        }
    }
}
