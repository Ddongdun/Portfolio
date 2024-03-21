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
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace KinectV01.UserInfoControls
{
    /// <summary>
    /// RankMainWindow.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class RankMainWindow : UserControl
    {
        public RankMainWindow()
        {
            InitializeComponent();
        }

        List<CelebPoint> celebPointComps = new List<CelebPoint>();
        public CelebPoint AddIdol(Idol idol)
        {
            var childCeleb = new CelebPoint();
            childCeleb.Width = double.NaN;
            childCeleb.VerticalAlignment = VerticalAlignment.Top;
            childCeleb.HorizontalAlignment = HorizontalAlignment.Stretch;
            childCeleb.lblIdolName.Content = idol.IName;
            childCeleb.lblIdolPoint.Content = idol.IScore;

            var userNames = idol.IdolUsers.Keys.ToList();
            var userPoints = idol.IdolUsers.Values.ToList();

            
            for(int i = 0; i<idol.IdolUsers.Count; i++)
            {
                childCeleb.AddGridChild(userNames[i], userPoints[i].ToString());
            }
            


            double top = 10;
            if (celebPointComps.Any())
            {
                var lastCeleb = celebPointComps.Last();
                top = lastCeleb.Margin.Top + lastCeleb.Height + 10;
            }

            childCeleb.Margin = new Thickness(0, top, 0, 0);
            gridList.Children.Add(childCeleb);
            celebPointComps.Add(childCeleb);
            childCeleb.btnExpand.Click += (s, args) => ToggleExpandCollapse(childCeleb, args);
            return childCeleb;
        }
        private void celebPointExpandButtonClicked(CelebPoint ParentSelebPoint, RoutedEventArgs e)
        {
            if (!celebPointComps.Any()) return;
            List<CelebPoint> belowCelebPoints = new List<CelebPoint>();
            foreach(CelebPoint celebPoint in celebPointComps)
            {
                if(celebPoint.Margin.Top > ParentSelebPoint.Margin.Top)
                {
                    belowCelebPoints.Add(celebPoint);
                }
            }
            foreach (var celebPoint in belowCelebPoints)
            {
                ThicknessAnimation thicknessAnimation = new ThicknessAnimation
                {
                    From = celebPoint.Margin,
                    To = new Thickness(celebPoint.Margin.Left, celebPoint.Margin.Top + 220, celebPoint.Margin.Right, celebPoint.Margin.Bottom),
                    Duration = new Duration(TimeSpan.FromSeconds(0.5)),
                    EasingFunction = new QuadraticEase() { EasingMode = EasingMode.EaseInOut }
                };

                celebPoint.BeginAnimation(FrameworkElement.MarginProperty, thicknessAnimation);
            }
        }
        private void celebPointCloseButtonClicked(CelebPoint ParentSelebPoint, RoutedEventArgs e)
        {
            if (!celebPointComps.Any()) return;
            List<CelebPoint> upperCelebPoint = new List<CelebPoint>();
            foreach (CelebPoint celebPoint in celebPointComps)
            {
                if (celebPoint.Margin.Top > ParentSelebPoint.Margin.Top)
                {
                    upperCelebPoint.Add(celebPoint);
                }
            }
            foreach (var celebPoint in upperCelebPoint)
            {
                ThicknessAnimation thicknessAnimation = new ThicknessAnimation
                {
                    From = celebPoint.Margin,
                    To = new Thickness(celebPoint.Margin.Left, celebPoint.Margin.Top - 220, celebPoint.Margin.Right, celebPoint.Margin.Bottom),
                    Duration = new Duration(TimeSpan.FromSeconds(0.5)),
                    EasingFunction = new QuadraticEase() { EasingMode = EasingMode.EaseInOut }
                };

                celebPoint.BeginAnimation(FrameworkElement.MarginProperty, thicknessAnimation);
            }
        }

        private void ToggleExpandCollapse(CelebPoint ParentSelebPoint, RoutedEventArgs e)
        {
            if (!ParentSelebPoint.IsExpanded)
            {
                celebPointCloseButtonClicked(ParentSelebPoint, e);
            }
            else
            {
                celebPointExpandButtonClicked(ParentSelebPoint, e);
            }
        }

        public void LoadPlayerInfo()
        {
            stackPanel.Children.Clear();
            var playerInfo = new PlayerDetailPoints();
            //playerInfo.lblPlayerName.Content = User.Instance.UName;
            //playerInfo.lblPoint.Content = User.Instance.IScore;
            //playerInfo.lblTime.Content = User.Instance.TotalPlayedTime.ToString();
            playerInfo.Margin = new Thickness(4, 4, 4, 4);
            stackPanel.Children.Add(playerInfo);
            
            var celebOfPlayer = new PlayerPointPerCeleb();
            celebOfPlayer.Margin = new Thickness(4, 4, 4, 4);
            stackPanel.Children.Add(celebOfPlayer);
            
        }

        public void reset()
        {
            this.gridList.Children.Clear();
            this.celebPointComps.Clear();
        }





        public void updateScoreAt(Idol idol,User user, int score)
        {
            try
            {
                this.Dispatcher.Invoke(() => // UI 스레드에서 실행
                {
                    //MessageBox.Show("updatescoreat 함수 시작");
                    var children = this.gridList.Children;
                    //if (children.Count == 0) MessageBox.Show("null값");

                    foreach (var child in children)
                    {
                        if (child is CelebPoint celebPoint && celebPoint.lblIdolName.Content.Equals(idol.IName))
                        {
                            celebPoint.lblIdolPoint.Content = score;
                            foreach(var innerchild in celebPoint.gridList.Children)
                            {
                                if (((CelebPointPlayerInfo)innerchild).lblUserName.Content.Equals(user.UName))
                                {
                                    ((CelebPointPlayerInfo)innerchild).lblUserPoint.Content = idol.IdolUsers[user.UName];
                                }
                            }
                            

                        }
                    }
                });
            }
            catch (Exception e)
            {
                MessageBox.Show(e.ToString());
            }
        }

    }
}
