using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
namespace KinectV01.UserInfoControls
{
    /// <summary>
    /// UserInfoDetail.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class UserInfoDetail : Window
    {
        private int progressBarValue = 0;
        public UserInfoDetail()
        {
            InitializeComponent();
            TextBoxTest.TextChanged += TextBox_TextChanged;//임시로 만든 textbox에 값을 입력하면(totalScore 대체)입력된 값에 따라 level표시.

        }
        private void TextBox_TextChanged(object sender, EventArgs e)
        {
            int number;
            if (int.TryParse(TextBoxTest.Text, out number))
            {
                if (number >= 100)
                {
                    ExpProgressBar.Value = number % 100;
                }
                else
                {
                    ExpProgressBar.Value = number;
                }
                UserLevellbl.Content = "User Level : " + (number / 100).ToString();
            }
        }
    }
}
