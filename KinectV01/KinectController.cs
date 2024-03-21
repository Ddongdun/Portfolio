using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using Microsoft.Kinect;
using System.Windows.Shapes;
using System.Windows.Input;
using System.Threading;
using System.Management;
using System.Windows.Threading;
using System.Runtime.InteropServices;
using KinectV01;
using System.Diagnostics;
using CefSharp.DevTools.Debugger;

namespace FirstPage
{
    internal class KinectController
    {
        private KinectSensor nui = null;
        private Dictionary<UIElement, Object> eventHandlers = new Dictionary<UIElement, Object>();


        /// <summary>
        /// 키넥트 생성자
        /// </summary>
        /// <param name="nui"></param>
        public KinectController()
        {
            try
            {
                nui = KinectSensor.KinectSensors.FirstOrDefault
                    (sensor => sensor.Status == KinectStatus.Connected);
                SetupUSBEventWatchers();

                nui.ColorStream.Enable();
                nui.DepthStream.Enable();
                nui.SkeletonStream.Enable();
                nui.Start();
            }
            catch
            {

            }
        }


        ////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// 키넥트 연결 해제 로직
        /// </summary>
        private void SetupUSBEventWatchers()
        {
            var insertQuery = new WqlEventQuery("SELECT * FROM __InstanceCreationEvent WITHIN 1 WHERE TargetInstance ISA 'Win32_USBHub'");
            var insertWatcher = new ManagementEventWatcher(insertQuery);
            var removeQuery = new WqlEventQuery("SELECT * FROM __InstanceDeletionEvent WITHIN 1 WHERE TargetInstance ISA 'Win32_USBHub'");
            var removeWatcher = new ManagementEventWatcher(removeQuery);

            bool insertWatcherFlag = true;
            bool removeWatcherFlag = true;

            insertWatcher.EventArrived += (sender, e) => Application.Current.Dispatcher.Invoke(() =>
            {
                if (insertWatcherFlag)
                {
                    insertWatcherFlag = false;
                    ReconnectKinect();
                    //Debug.WriteLine("insert called");
                }
                removeWatcherFlag = true;

            });

            removeWatcher.EventArrived += (sender, e) => Application.Current.Dispatcher.Invoke(() =>
            {
                if (removeWatcherFlag)
                {
                    removeWatcherFlag = false;
                    MessageBox.Show("키넥트 해제됨");
                    //Debug.WriteLine("remove called");
                    //토큰 취소
                    ClearEventHandler(lastImage);
                    nui.ColorStream.Disable();
                    nui.Stop();
                }
                insertWatcherFlag = true;
            });

            removeWatcher.Start();
            insertWatcher.Start();
        }

        private async void ReconnectKinect()
        {
                await Task.Run(() =>
                {
                    while (KinectSensor.KinectSensors.Count == 0)
                    {
                        Task.Delay(1000);
                    }
                    while (true)
                    {
                        nui = KinectSensor.KinectSensors.FirstOrDefault
                        (sensor => sensor.Status == KinectStatus.Connected);
                        if(nui != null)
                        {
                            break;
                        }
                    }
                    
                });

            MessageBox.Show("키넥트 다시 연결됨");
            nui.Start();
            nui.ColorStream.Enable();
            DisplayColorStreamAt(lastImage);
        }

        ////////////////////////////////////////////////////////////////////////





        /// <summary>
        /// 키넥트 화면 출력
        /// </summary>
        /// <param name="image"></param>
        /// 
        public Image lastImage = null;
        public EventHandler<Microsoft.Kinect.ColorImageFrameReadyEventArgs> ColorEventHandler = null;
        public void DisplayColorStreamAt(Image image)
        {
            if (nui == null) return;
            if (nui.ColorStream == null) nui.ColorStream.Enable();

            EventHandler<Microsoft.Kinect.ColorImageFrameReadyEventArgs> handler = null;

            handler = (sender, e) =>
            {
                ColorImageFrame ImageParam = e.OpenColorImageFrame();
                if (ImageParam == null) return;
                byte[] ImageBits = new byte[ImageParam.PixelDataLength];
                ImageParam.CopyPixelDataTo(ImageBits);
                BitmapSource src = null;
                src = BitmapSource.Create(ImageParam.Width,
                ImageParam.Height,
                96, 96,
                PixelFormats.Bgr32,
                null,
                ImageBits,
                ImageParam.Width * ImageParam.BytesPerPixel);


                Application.Current.Dispatcher.Invoke(() =>
                {
                    image.Source = src;

                });
            };

            nui.ColorFrameReady += handler;
            ColorEventHandler = handler;
        }


        public void DisplayDepthStreamAt(Image image)
        {
            if (nui == null) return;
            if (nui.DepthStream == null) nui.DepthStream.Enable();

            EventHandler<Microsoft.Kinect.DepthImageFrameReadyEventArgs> handler = null;

            handler = (sender, e) =>
            {
                DepthImageFrame ImageParam = e.OpenDepthImageFrame();
                if (ImageParam == null) return;
                short[] ImageBits = new short[ImageParam.PixelDataLength];
                ImageParam.CopyPixelDataTo(ImageBits);
                BitmapSource src = null;
                src = BitmapSource.Create(ImageParam.Width,
                ImageParam.Height,
                96, 96,
                PixelFormats.Gray16,
                null,
                ImageBits,
                ImageParam.Width * ImageParam.BytesPerPixel);
                image.Source = src;
            };

            nui.DepthFrameReady += handler;
            eventHandlers.Add(image, handler);
        }



        public void DisplaySkeletonAt(Canvas canvas, Image image)
        {
            MessageBox.Show(nui.Status.ToString());
            nui.SkeletonStream.Enable();
            nui.DepthStream.Enable();

            Polyline[] m_poly = new Polyline[5];

            for (int i = 0; i < 5; i++)
            {
                m_poly[i] = new Polyline();
                m_poly[i].Stroke = new SolidColorBrush(Colors.Green);
                m_poly[i].StrokeThickness = 4;
                m_poly[i].Visibility = Visibility.Collapsed;
                Canvas.SetTop(m_poly[i], 0);
                Canvas.SetLeft(m_poly[i], 0);
                canvas.Children.Add(m_poly[i]);
            }

            EventHandler<Microsoft.Kinect.AllFramesReadyEventArgs> handler = null;

            handler = (sender, e) =>
            {
                try
                {
                    SkeletonFrame sf = e.OpenSkeletonFrame();
                    if (sf == null) return;
                    Skeleton[] skeletonData = new Skeleton[sf.SkeletonArrayLength];
                    sf.CopySkeletonDataTo(skeletonData);
                    using (DepthImageFrame depthImageFrame = e.OpenDepthImageFrame())
                    {
                        if (depthImageFrame != null)
                        {
                            foreach (Skeleton sd in skeletonData)
                            {
                                if (sd.TrackingState == SkeletonTrackingState.Tracked)
                                {
                                    MessageBox.Show("스켈레톤 인식됨");
                                    Debug.WriteLine("스켈레톤 인식됨");
                                    int nMax = 20;
                                    Joint[] joints = new Joint[nMax];
                                    for (int j = 0; j < nMax; j++)
                                    {
                                        joints[j] = sd.Joints[(JointType)j];
                                    }
                                    Point[] points = new Point[nMax];
                                    for (int j = 0; j < nMax; j++)
                                    {
                                        DepthImagePoint depthPoint;
                                        depthPoint = depthImageFrame.MapFromSkeletonPoint(joints[j].Position);
                                        points[j] = new Point((int)(image.Width *
                                        depthPoint.X / depthImageFrame.Width),
                                        (int)(image.Height *
                                        depthPoint.Y / depthImageFrame.Height));
                                    }
                                    PointCollection pc0 = new PointCollection(4);
                                    pc0.Add(points[(int)JointType.HipCenter]);
                                    pc0.Add(points[(int)JointType.Spine]);
                                    pc0.Add(points[(int)JointType.ShoulderCenter]);
                                    pc0.Add(points[(int)JointType.Head]);
                                    m_poly[0].Points = pc0;
                                    m_poly[0].Visibility = Visibility.Visible;

                                    PointCollection pc1 = new PointCollection(5);
                                    pc1.Add(points[(int)JointType.ShoulderCenter]);
                                    pc1.Add(points[(int)JointType.ShoulderLeft]);
                                    pc1.Add(points[(int)JointType.ElbowLeft]);
                                    pc1.Add(points[(int)JointType.WristLeft]);
                                    pc1.Add(points[(int)JointType.HandLeft]);
                                    m_poly[1].Points = pc1;
                                    m_poly[1].Visibility = Visibility.Visible;

                                    PointCollection pc2 = new PointCollection(5);
                                    pc2.Add(points[(int)JointType.ShoulderCenter]);
                                    pc2.Add(points[(int)JointType.ShoulderRight]);
                                    pc2.Add(points[(int)JointType.ElbowRight]);
                                    pc2.Add(points[(int)JointType.WristRight]);
                                    pc2.Add(points[(int)JointType.HandRight]);
                                    m_poly[2].Points = pc2;
                                    m_poly[2].Visibility = Visibility.Visible;

                                    PointCollection pc3 = new PointCollection(5);
                                    pc3.Add(points[(int)JointType.HipCenter]);
                                    pc3.Add(points[(int)JointType.HipLeft]);
                                    pc3.Add(points[(int)JointType.KneeLeft]);
                                    pc3.Add(points[(int)JointType.AnkleLeft]);
                                    pc3.Add(points[(int)JointType.FootLeft]);
                                    m_poly[3].Points = pc3;
                                    m_poly[3].Visibility = Visibility.Visible;

                                    PointCollection pc4 = new PointCollection(5);
                                    pc4.Add(points[(int)JointType.HipCenter]);
                                    pc4.Add(points[(int)JointType.HipRight]);
                                    pc4.Add(points[(int)JointType.KneeRight]);
                                    pc4.Add(points[(int)JointType.AnkleRight]);
                                    pc4.Add(points[(int)JointType.FootRight]);
                                    m_poly[4].Points = pc4;
                                    m_poly[4].Visibility = Visibility.Visible;
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }

            };

            nui.AllFramesReady += handler;
        }

        /// <summary>
        /// 키넥트 출력 정지
        /// </summary>
        /// <param name="element"></param>
        public void ClearEventHandler(UIElement element)
        {
            if (nui != null && eventHandlers.TryGetValue(element, out var value))
            {
                if (value is EventHandler<Microsoft.Kinect.DepthImageFrameReadyEventArgs> depthHandler)
                {
                    nui.DepthFrameReady -= depthHandler;
                    eventHandlers.Remove(element); // 딕셔너리에서 항목 제거
                }
                if (value is EventHandler<Microsoft.Kinect.ColorImageFrameReadyEventArgs> colorHandler)
                {
                    nui.ColorFrameReady -= colorHandler;
                    eventHandlers.Remove(element); // 딕셔너리에서 항목 제거
                }
                if (value is EventHandler<Microsoft.Kinect.AllFramesReadyEventArgs> allFramesHandler)
                {
                    nui.AllFramesReady -= allFramesHandler;
                    eventHandlers.Remove(element); // 딕셔너리에서 항목 제거
                }
            }
        }

        public void StopDisplayColorFrame()
        {
            try
            {
                nui.ColorFrameReady -= ColorEventHandler;

            }
            catch
            {

            }
        }

        /// <summary>
        /// 키넥트 움직임 감지 로직
        /// </summary>
        /// <param name="callback"></param>
        /// <param name="tSource"></param>
        //cancellationTokenSource?.Cancel();
        public void StartCapturingPlayerMovement(Action<double> callback, out CancellationTokenSource tSource)
        {
            tSource = new CancellationTokenSource();
            CancellationToken cToken = tSource.Token;

            Skeleton prevSkeleton = null;
            bool isFirstFrame = true;

            Task.Run(() =>
            {
                while (!cToken.IsCancellationRequested)
                {
                    using (SkeletonFrame sf = nui.SkeletonStream.OpenNextFrame(100))
                    {
                        if (sf != null)
                        {
                            Skeleton[] skeletons = new Skeleton[sf.SkeletonArrayLength];
                            sf.CopySkeletonDataTo(skeletons);
                            Skeleton skeleton = skeletons.FirstOrDefault(s => s.TrackingState == SkeletonTrackingState.Tracked);

                            if (skeleton != null)
                            {
                                if (!isFirstFrame)
                                {
                                    Joint rightHand = skeleton.Joints[JointType.HandRight];
                                    Joint prevRightHand = prevSkeleton.Joints[JointType.HandRight];
                                    Joint head = skeleton.Joints[JointType.Head];
                                    Joint hip = skeleton.Joints[JointType.HipCenter];

                                    double deltaX = rightHand.Position.X - prevRightHand.Position.X;
                                    double deltaY = rightHand.Position.Y - prevRightHand.Position.Y;
                                    double deltaZ = rightHand.Position.Z - prevRightHand.Position.Z;


                                    double score = Math.Sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
                                    //score = ScaleScoreByDistance(humanScale, score);
                                    callback(score);

                                }
                                else
                                {
                                    isFirstFrame = false;
                                }
                                prevSkeleton = skeleton;
                            }
                        }
                    }
                    Task.Delay(100);
                }
            }, cToken);
        }

        ////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// 키넥트 마우스 커서 움직임 구현
        /// </summary>
        /// <param name="X"></param>
        /// <param name="Y"></param>
        /// <returns></returns>
        [DllImport("user32.dll")]
        public static extern bool SetCursorPos(int X, int Y);
        [DllImport("user32.dll")]
        static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, int dwExtraInfo);

        private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        private const uint MOUSEEVENTF_LEFTUP = 0x0004;



        public void MoveCursor(MainWindow window, int deltaX, int deltaY)
        {
            window.Dispatcher.Invoke(() =>
            {
                int windowLeft = (int)window.Left;
                int windowTop = (int)window.Top;
                int windowRight = (int)(window.Left + window.Width);
                int windowBottom = (int)(window.Top + window.Height);

                System.Drawing.Point currentPos = System.Windows.Forms.Cursor.Position;

                int newX = currentPos.X + deltaX;
                int newY = currentPos.Y + deltaY;

                newX = Math.Max(windowLeft, Math.Min(newX, windowRight + 1000));
                newY = Math.Max(windowTop, Math.Min(newY, windowBottom + 1000));

                SetCursorPos(newX, newY);
            });
        }


        public void CalcAndMoveCursor(MainWindow window, out CancellationTokenSource tSource)
        {
            tSource = new CancellationTokenSource();
            CancellationToken cToken = tSource.Token;

            Skeleton prevSkeleton = null;
            bool isFirstFrame = true;
            bool isAlreadyClicked = false;   //이 플래그를 사용

            Task.Run(() =>
            {
                while (!cToken.IsCancellationRequested)
                {
                    using (SkeletonFrame sf = nui.SkeletonStream.OpenNextFrame(100))
                    {
                        if (sf != null)
                        {
                            Skeleton[] skeletons = new Skeleton[sf.SkeletonArrayLength];
                            sf.CopySkeletonDataTo(skeletons);
                            Skeleton skeleton = skeletons.FirstOrDefault(s => s.TrackingState == SkeletonTrackingState.Tracked);

                            if (skeleton != null)
                            {
                                if (!isFirstFrame)
                                {
                                    Joint rightHand = skeleton.Joints[JointType.HandRight];
                                    Joint prevRightHand = prevSkeleton.Joints[JointType.HandRight];

                                    //머리와 왼손의 위치 추적?

                                    int deltaX = (int)((rightHand.Position.X - prevRightHand.Position.X) * 2500); // 스케일 조정 필요
                                    int deltaY = (int)((rightHand.Position.Y - prevRightHand.Position.Y) * 2500); // 스케일 조정 필요

                                    window.Dispatcher.Invoke(() => MoveCursor(window, deltaX, -deltaY));

                                    Joint head = skeleton.Joints[JointType.Head];
                                    Joint handLeft = skeleton.Joints[JointType.HandLeft];

                                    if (handLeft.Position.Y > head.Position.Y && !isAlreadyClicked)
                                    {
                                        window.Dispatcher.Invoke(ClickCursor);
                                        isAlreadyClicked = true;
                                    }
                                    else if (handLeft.Position.Y <= head.Position.Y)
                                    {
                                        isAlreadyClicked = false;
                                    }

                                }
                                else
                                {
                                    isFirstFrame = false;
                                }
                                prevSkeleton = skeleton;
                            }
                        }
                    }
                    Thread.Sleep(30);
                }
            }, cToken);
        }

        private void ClickCursor()
        {
            // 마우스 왼쪽 버튼 클릭
            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
        }



        ///////////////////////////////////////////////////////////////////////////////////////

        public void DisplayHandAndCalcPoint(Action<double> callback, Canvas canvas,Image image, out CancellationTokenSource tSource)
        {
            tSource = new CancellationTokenSource();
            CancellationToken cToken = tSource.Token;


            Skeleton prevSkeleton = null;
            Ellipse[] prevEllipse = new Ellipse[2];
            bool isFirstFrame = true;

            Task.Run(() =>
            {
                while (!cToken.IsCancellationRequested)
                {
                    using (SkeletonFrame sf = nui.SkeletonStream.OpenNextFrame(100))
                    {
                        if (sf != null)
                        {
                            Skeleton[] skeletons = new Skeleton[sf.SkeletonArrayLength];
                            sf.CopySkeletonDataTo(skeletons);
                            Skeleton skeleton = skeletons.FirstOrDefault(s => s.TrackingState == SkeletonTrackingState.Tracked);

                            if (skeleton != null)
                            {
                                if (!isFirstFrame)
                                {
                                    Joint rightHand = skeleton.Joints[JointType.HandRight];
                                    Joint leftHand = skeleton.Joints[JointType.HandLeft];
                                    Joint prevRightHand = prevSkeleton.Joints[JointType.HandRight];
                                    Joint prevLeftHand = prevSkeleton.Joints[JointType.HandLeft];

                                    Joint head = skeleton.Joints[JointType.Head];
                                    Joint hip = skeleton.Joints[JointType.HipCenter];

                                    double deltaX = rightHand.Position.X - prevRightHand.Position.X;
                                    double deltaY = rightHand.Position.Y - prevRightHand.Position.Y;
                                    double deltaZ = rightHand.Position.Z - prevRightHand.Position.Z;


                                    double score = Math.Sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
                                    //score = ScaleScoreByDistance(humanScale, score);
                                    if(score < 0)
                                    {
                                        continue;
                                    }

                                    callback(score);


                                    Application.Current.Dispatcher.Invoke(() =>
                                    {
                                        if (prevEllipse[0] != null)
                                        {
                                            canvas.Children.Remove(prevEllipse[0]);
                                        }
                                        if (prevEllipse[1] != null)
                                        {
                                            canvas.Children.Remove(prevEllipse[1]);
                                        }
                                        if (!cToken.IsCancellationRequested)
                                        {
                                            prevEllipse[0] = DrawHandCircle(rightHand.Position, canvas);
                                            prevEllipse[1] = DrawHandCircle(leftHand.Position, canvas);
                                        }

                                    });
                                    


                                }
                                else
                                {
                                    isFirstFrame = false;
                                }
                                prevSkeleton = skeleton;
                            }
                        }
                    }
                    Task.Delay(100);
                }
            }, cToken);
        }

        private Ellipse DrawHandCircle(SkeletonPoint handPoint, Canvas canvas)
        {
            CoordinateMapper mapper = nui.CoordinateMapper;
            var mappedPoint = mapper.MapSkeletonPointToColorPoint(handPoint, ColorImageFormat.RgbResolution640x480Fps30);


            var handCircle = new Ellipse
            {
                Width = 30,
                Height = 30,
                Fill = new SolidColorBrush(Colors.Red),
                Stroke = new SolidColorBrush(Colors.Black),
                StrokeThickness = 2
            };

            double xScale = canvas.ActualWidth / 640;
            double yScale = canvas.ActualHeight / 480;

            var canvasX = mappedPoint.X * xScale;
            var canvasY = mappedPoint.Y * yScale;

            canvasX = Math.Max(0, Math.Min(canvasX, canvas.ActualWidth - handCircle.Width));
            canvasY = Math.Max(0, Math.Min(canvasY, canvas.ActualHeight - handCircle.Height));



            Canvas.SetLeft(handCircle, canvasX - handCircle.Width / 2);
            Canvas.SetTop(handCircle, canvasY - handCircle.Height / 2);
            canvas.Children.Add(handCircle);
            return handCircle;
        }


    }


}
