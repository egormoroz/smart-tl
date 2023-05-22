using Microsoft.AspNetCore.Http.Extensions;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Drawing;
using System.IO;
using System.Net.Sockets;
using System.Reflection;
using System.Security.Principal;
using System.Text;
using System.Text.Json;

namespace TrafficLight.Pages
{
    public class IndexModel : PageModel
    {
        public int Amount { get; set; }
        public List<Car>? Cars { get; set; }
        public Canvas Canvas { get; set; }
        public byte phase { get; set; }
        public static byte[] responseData = new byte[512];
        public IndexModel()
        {
            Cars = new List<Car>();
            Canvas = new Canvas();
            StringBuilder response = Server();
            string carsSBjson = response.ToString().Trim('\0');
            Message? message = JsonSerializer.Deserialize<Message>(carsSBjson);
            Canvas.w = message.Canvas.w;
            Canvas.h = message.Canvas.h;

            for (int i = 0; i < message.Cars.Count; i++)
            {
                Cars.Add(message.Cars[i]);
            }
            

            phase = message.Phase;

            Amount = Cars.Count;

            List<Color> colors = new List<Color>() { Color.Aquamarine, Color.Crimson, Color.GreenYellow, Color.Indigo };
            for(int i = 0; i < Cars.Count; ++i)
            {
                Cars[i].Color = colors[i % colors.Count];
            }
        }

        private static StringBuilder Server()
        {
            const string ip = "localhost";
            const int port = 8888;
            TcpClient tcpClient = new TcpClient();
            tcpClient.Connect(ip, port);
            NetworkStream networkStream = tcpClient.GetStream();
            // string requestMessage = $"GET \r\nHost: {ip}\r\n";
            string requestMessage = "client".PadRight(32, '\0');
            var requestData = Encoding.ASCII.GetBytes(requestMessage);
            networkStream.Write(requestData);

            // буфер для получения данных
            
            // StringBuilder для склеивания полученных данных в одну строку
            var response = new StringBuilder();
            int bytes;  // количество полученных байтов
            do
            {
                // получаем данные
                bytes = networkStream.Read(responseData);
                // преобразуем в строку и добавляем ее в StringBuilder
                response.Append(Encoding.ASCII.GetString(responseData, 0, bytes));
            }
            while (bytes > 0); // пока данные есть в потоке 

            // выводим данные на консоль
            //Console.WriteLine(response);
            return response;
        }

        public void OnGet()
        {
        }

    }
}