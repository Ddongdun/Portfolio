using CefSharp.DevTools.IndexedDB;
using KinectV01.args;
using KinectV01.UserInfoControls;
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
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace KinectV01
{
    /// <summary>
    /// MainWindow.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            var viewModel = (MainViewModel)Application.Current.TryFindResource("mainViewModel");
            viewModel.UpdateRank += updateComponents;
            viewModel.UpdateUiScore += updateComponentsScore;
            viewModel.UpdateURLEvent += updateURL;
            viewModel.ResetProgressBarEvent += ResetProgressBar;
        }

        private void ResetProgressBar(object sender, EventArgs e)
        {
            scoreProgressBar.UpdateProgressBar(0);
            
            scoreStack = 0;

        
                for (int i = canvas01.Children.Count - 1; i >= 0; i--)
                {
                    var child = canvas01.Children[i];
                    if (child is Ellipse)
                    {
                        canvas01.Children.RemoveAt(i);
                    }
                }
            
            Task.Delay(500);
            image01.Source = null;
            canvas01.InvalidateVisual();
            

         

        }

        private void updateURL(object sender, UpdateURLArgs e)
        {
            //scoreProgressBar.UpdateProgressBar(0);
            browser.Load(e.URL);
        }

        private void updateComponents(object sender, args.UpdateRankArgs e)
        {
            rankWindow.reset();
            foreach(var idol in e.Idols)
            {
                rankWindow.AddIdol(idol);
            }
        }

        int scoreStack = 0;
        private void updateComponentsScore(object sender, args.UpdateUiScoreArgs e)
        {

            Application.Current.Dispatcher.Invoke(() =>
            {
                rankWindow.updateScoreAt(e.Idol, e.User, e.Idol.IScore);
                if (e.Score>0.01)//여기에 원하는 조건식 추가
                {
                    scoreStack += e.Score;
                    scoreProgressBar.UpdateProgressBar(scoreStack);
                }
            });

            
        }


        private void UserInfoBtn_Click(object sender, RoutedEventArgs e)
        {
            UserInfoDetail abc = new UserInfoDetail();
            abc.Show();
        }

        private void rankWindow_Loaded(object sender, RoutedEventArgs e)
        {

        }
    }
}
