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
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace KinectV01.UserInfoControls
{
    /// <summary>
    /// CelebPoint.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class CelebPoint : UserControl
    {
        public bool IsExpanded { get; set; } = false;
        public CelebPoint()
        {
            InitializeComponent();
            this.Height = 80;

        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            DoubleAnimation heightAnimation = new DoubleAnimation
            {
                Duration = new Duration(TimeSpan.FromSeconds(0.5)),
                AutoReverse = false,
                EasingFunction = new QuadraticEase() { EasingMode = EasingMode.EaseInOut }
            };

            if (IsExpanded)
            {
                heightAnimation.From = this.Height;
                heightAnimation.To = 80;
            }
            else
            {
                heightAnimation.From = this.Height;
                heightAnimation.To = 300;
            }

            this.BeginAnimation(FrameworkElement.HeightProperty, heightAnimation);
            if(!IsExpanded)
            {
                btnExpand.Content = "Minimize";
            }
            else
            {
                btnExpand.Content = "Expand";
            }
            IsExpanded = !IsExpanded;
        }


        List<CelebPointPlayerInfo> celebPointPlayerInfos = new List<CelebPointPlayerInfo>();
        public void AddGridChild(string name, string point)
        {
            CelebPointPlayerInfo celebPointPlayerInfo = new CelebPointPlayerInfo();
            celebPointPlayerInfos.Add(celebPointPlayerInfo);
            celebPointPlayerInfo.lblUserName.Content = name;
            celebPointPlayerInfo.lblUserPoint.Content = point;
            double topMargin = (celebPointPlayerInfos.Count - 1) * (celebPointPlayerInfo.Height + 4) + 4;
            celebPointPlayerInfo.VerticalAlignment = VerticalAlignment.Top;
            celebPointPlayerInfo.HorizontalAlignment = HorizontalAlignment.Stretch;
            celebPointPlayerInfo.Margin = new Thickness(0, topMargin, 0, 0);
            gridList.Children.Add(celebPointPlayerInfo);
        }

        private void UpdatePlayerRankings()
        {
            // 1. 점수에 따라 리스트 정렬 (예시: Score 프로퍼티를 기준으로 내림차순)
            celebPointPlayerInfos.Sort((a, b) => b.Score.CompareTo(a.Score));

            // 2. 각 객체의 랭킹에 따른 margin 계산
            for (int i = 0; i < celebPointPlayerInfos.Count; i++)
            {
                double topMargin = i * (celebPointPlayerInfos[i].Height + 4) + 4;

                // 3. 애니메이션으로 margin 변경
                ThicknessAnimation marginAnimation = new ThicknessAnimation
                {
                    From = celebPointPlayerInfos[i].Margin,
                    To = new Thickness(0, topMargin, 0, 0),
                    Duration = new Duration(TimeSpan.FromSeconds(0.5)),
                    EasingFunction = new QuadraticEase() { EasingMode = EasingMode.EaseInOut }
                };

                celebPointPlayerInfos[i].BeginAnimation(MarginProperty, marginAnimation);
            }
        }

        private Random rnd = new Random();

        private void RandomizeScore()
        {
            foreach (var player in celebPointPlayerInfos)
            {
                player.Score = rnd.Next(0, 101);
            }
        }

        private void btnUpdateRank_Click(object sender, RoutedEventArgs e)
        {
            UpdatePlayerRankings();
        }

        private void btnRandomizeScore_Click(object sender, RoutedEventArgs e)
        {
            RandomizeScore();
            UpdatePlayerRankings();

        }
    }
}
