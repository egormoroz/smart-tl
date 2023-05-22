using System.Text.Json.Serialization;

namespace TrafficLight
{
    public class Message
    {
        [JsonPropertyName("cars")]
        public List<Car> Cars { get; set; }
        [JsonPropertyName("canvas")]
        public Canvas Canvas { get; set; }
        [JsonPropertyName("phase")]
        public byte Phase { get; set; }
    }
    public class Canvas
    {
        public double w { get; set; }
        public double h { get; set; }
    }
}
