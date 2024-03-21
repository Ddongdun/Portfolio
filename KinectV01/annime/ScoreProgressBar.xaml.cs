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
using System.Windows.Threading;

namespace KinectV01.annime
{
    /// <summary>
    /// ScoreProgressBar.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class ScoreProgressBar : UserControl
    {
        private DispatcherTimer timer;
        private int progressBarValue = 0;
        private DoubleAnimation animation; // 여기에 animation 변수를 선언합니다.
        Random random;


        public ScoreProgressBar()
        {
            InitializeComponent();

            // 애니메이션 생성
            animation = new DoubleAnimation();
            animation.AutoReverse = true;
            animation.RepeatBehavior = RepeatBehavior.Forever;
            random = new Random();
            TextBox.TextChanged += TextBox_TextChanged;

        }
        private void TextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            // TextBox에 입력된 값 가져오기
            if (int.TryParse(TextBox.Text, out int inputValue))
            {
                // 입력된 값으로 ProgressBar 업데이트
                UpdateProgressBar(inputValue);
            }
        }
        public void UpdateProgressBar(int value)
        {
            try
            {
                progressBarValue = value;
                if (progressBarValue <= 1000)
                {
                    ProgressBar1.Value = progressBarValue;

                    // 10% 증가할 때 마다 점수 출력
                    if (progressBarValue % 1 == 0)
                    {
                        //DisplayNumber(progressBarValue);
                    }
                }
                else
                {
                    ProgressBar1.Value = 1000;
                }
            }
            catch(Exception e)
            {
                MessageBox.Show(e.ToString());
            }

        }

        private void DisplayNumber(int number)
        {
            int randomValue = random.Next(-100, 101);
            NumberTextBlock.Text = number.ToString();
            NumberTextBlock.Visibility = Visibility.Visible;

            // 초기 위치 설정
            NumberTextBlock.RenderTransform = new TranslateTransform(randomValue, 100);

            // 애니메이션 생성
            DoubleAnimation translateYAnimation = new DoubleAnimation
            {
                To = 0,
                Duration = TimeSpan.FromSeconds(1) // 애니메이션 지속 시간
            };

            // 애니메이션의 대상과 속성 설정
            TranslateTransform translateTransform = NumberTextBlock.RenderTransform as TranslateTransform;
            if (translateTransform != null)
            {
                translateTransform.BeginAnimation(TranslateTransform.YProperty, translateYAnimation);
            }

            // 타이머 설정
            var countdownTimer = new DispatcherTimer();
            countdownTimer.Interval = TimeSpan.FromSeconds(2); // 2초 동안 표시
            countdownTimer.Tick += (s, e) =>
            {
                NumberTextBlock.Visibility = Visibility.Hidden;
                countdownTimer.Stop();
            };
            countdownTimer.Start();
        }

        private void ProgressBar_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {

            double progress = ProgressBar1.Value;

            // 만약 ProgressBar의 값이 20% 이상이면 crowd_image를 보이도록 설정
            if (progress >= 200 && progress<500)
            {
                crowd_image.Visibility = Visibility.Visible;
                animation.To = 5; // TranslateTransform.Y
                animation.Duration = TimeSpan.FromSeconds(0.5);
                animation.EasingFunction = new BounceEase { Bounces = 5, Bounciness = 10, EasingMode = EasingMode.EaseInOut };
                // 애니메이션 적용
                TranslateTransform transform = ((TransformGroup)crowd_image.RenderTransform).Children[3] as TranslateTransform;
                transform.BeginAnimation(TranslateTransform.YProperty, animation);
            }

            else if (progress >= 500 && progress <700)
            {
                var countdownTimer1 = new DispatcherTimer();
                countdownTimer1.Interval = TimeSpan.FromSeconds(0.2);
                crowd_image.Visibility = Visibility.Hidden;
                crowd_image_glow_stick_1.Visibility = Visibility.Visible;
                crowd_image_glow_stick_2.Visibility = Visibility.Visible;
                // ProgressBar의 값이 51~100일 때
                animation.To = 20; // TranslateTransform.Y
                animation.Duration = TimeSpan.FromSeconds(0.2);
                animation.EasingFunction = new BounceEase { Bounces = 10, Bounciness = 20, EasingMode = EasingMode.EaseInOut };
                TranslateTransform transform = ((TransformGroup)crowd_image_glow_stick_1.RenderTransform).Children[3] as TranslateTransform;
                transform.BeginAnimation(TranslateTransform.YProperty, animation);
                TranslateTransform transform2 = ((TransformGroup)crowd_image_glow_stick_2.RenderTransform).Children[3] as TranslateTransform;
                transform2.BeginAnimation(TranslateTransform.YProperty, animation);
                // Tick 이벤트를 추가합니다.
                countdownTimer1.Tick += (s1, e1) =>
                {
                    // crowd_image_glow_stick_1과 crowd_image_glow_stick_2의 Visibility를 번갈아 가며 변경합니다.
                    if (crowd_image_glow_stick_1.Visibility == Visibility.Visible)
                    {
                        crowd_image_glow_stick_1.Visibility = Visibility.Hidden;
                        crowd_image_glow_stick_2.Visibility = Visibility.Visible;
                    }
                    else
                    {
                        crowd_image_glow_stick_1.Visibility = Visibility.Visible;
                        crowd_image_glow_stick_2.Visibility = Visibility.Hidden;
                    }
                };
                // Timer를 시작합니다.
                countdownTimer1.Start();

            }
            if (progress >= 700 && progress<=1000)
            {
                light_left.Visibility = Visibility.Visible;
                light_right.Visibility = Visibility.Visible;

                // light_left 이미지에 RotateTransform 적용
                var rotateTransformLeft = new RotateTransform(0);
                light_left.RenderTransform = rotateTransformLeft;

                // light_right 이미지에 RotateTransform 적용
                var rotateTransformRight = new RotateTransform(0);
                light_right.RenderTransform = rotateTransformRight;

                // 애니메이션 생성
                var animation = new DoubleAnimation
                {
                    To = 15, // 15도 회전
                    Duration = TimeSpan.FromSeconds(2), // 1초 동안
                    AutoReverse = true, // 자동으로 원래 상태로 복귀
                    RepeatBehavior = RepeatBehavior.Forever, // 무한 반복
                };

                // light_left 이미지에 애니메이션 적용
                rotateTransformLeft.BeginAnimation(RotateTransform.AngleProperty, animation);

                // light_right 이미지에 애니메이션 적용 (반시계 방향으로 회전)
                animation.To = -15; // -15도 회전 (반시계 방향)
                rotateTransformRight.BeginAnimation(RotateTransform.AngleProperty, animation);
            }
        }

    }
}
    
