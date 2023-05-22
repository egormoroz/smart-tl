using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace TrafficLight.Pages
{
    public class TestModel : PageModel
    {
        public string? Message { get; set; }
        [BindProperty]
        public string? Name { get; set; }
        [BindProperty]
        public int Age { get; set; }
        public void OnGet()
        {
            Message = "������� ������";
        }
        public void OnPost()
        {
            Message = $"���: {Name}\n�������: {Age}";
        }
    }
}

