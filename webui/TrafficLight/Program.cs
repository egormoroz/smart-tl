using Microsoft.AspNetCore.Hosting.Server;
using System.ComponentModel;
using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace TrafficLight
{
    public class Program
    {
        public static List<Car>? Cars {  get; set; }
        public static async Task Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);
            builder.Services.AddRazorPages();
            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (!app.Environment.IsDevelopment())
            {
                app.UseExceptionHandler("/Error");
                // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
                app.UseHsts();
            }

            //string ip = "192.168.31.36";
            //int port = 8888;
            //TcpClient tcpClient = new TcpClient();
            //await tcpClient.ConnectAsync(ip, port);
            //NetworkStream networkStream = tcpClient.GetStream();
            //string requestMessage = $"GET / HTTP/1.1\r\nHost: {ip}\r\nConnection: Close\r\n\r\n";
            //var requestData = Encoding.ASCII.GetBytes(requestMessage);
            //await networkStream.WriteAsync(requestData);

            //var responseData = new byte[1024];
            //var response = new StringBuilder();
            //int bytes;  // количество полученных байтов
            //do
            //{
            //    bytes = await networkStream.ReadAsync(responseData);
            //    response.Append(Encoding.ASCII.GetString(responseData, 0, bytes));
            //}
            //while (bytes > 0);

            //Console.WriteLine(response);
            //string carsSBjson = response.ToString().Trim('\0');
            


            app.UseHttpsRedirection();
            app.UseStaticFiles();

            app.UseRouting();

            app.MapRazorPages();

            app.Run();
            

        }
    }
}