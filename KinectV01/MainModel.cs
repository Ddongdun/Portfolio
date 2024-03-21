using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Forms;
using CefSharp.DevTools.Audits;
using FirstPage;
using Microsoft.Kinect;

namespace KinectV01
{
    public class MainModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        private List<Idol> Idols = null;
        private List<User> users = null;

        User currentUser = null;
        Idol currentIdol = null;
        int currentScore = 0;



        public MainModel()
        {
            this.Idols = DB.getIdolsFromDB();
            this.users = DB.getUserFromDB();
        }

        public List<Idol> GetIdols()
        {
            return this.Idols;
        }


        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }











        ///<summary>
        ///키넥트 관련
        ///</summary>


        KinectController kinectController = new KinectController();
        public void startPointCalc(Canvas canvas, Image image)
        {
            MessageBox.Show("측정 시작");
            kinectController.DisplayColorStreamAt(image);
            kinectController.DisplayHandAndCalcPoint((double score) =>
            {
                currentIdol.IScore += (int)(score*100);
                currentIdol.IdolUsers[currentUser.UName] += (int)(score * 100);

                currentUser.UScore += (int)(score * 100);
                currentUser.UserIdols[currentIdol.IName] += (int)(score * 100);

                currentScore += (int)(score * 100);

                UpdateUiScore?.Invoke(this, new args.UpdateUiScoreArgs(this.currentIdol, this.currentUser, (int)(score*20)));

            }, canvas, image, out StopCalcTokenSource);

        }

        public event EventHandler<args.UpdateUiScoreArgs> UpdateUiScore;
        CancellationTokenSource StopCalcTokenSource = null;

        ///<summary>
        ///DB 관련
        /// </summary>

        public void startDisplayColorStream(object imageObj)
        {
            Image targetImage = imageObj as Image;
            kinectController.DisplayColorStreamAt(targetImage);
        }

        public void startSkeletonStream(object canvasObj, object imageObj)
        {
            Canvas targetCanvas = canvasObj as Canvas;
            Image targetImage = imageObj as Image;
            kinectController.DisplaySkeletonAt(targetCanvas, targetImage);
        }




        DB DB = new DB();
        public List<Idol> getIdolAndUserFromDB()
        {
            var idols = DB.getIdolsFromDB();
            var users = DB.getUserFromDB();


            
            foreach(var idol in idols)
            {
                this.Idols.Add(idol);
            }

            foreach (var user in users)
            {
                this.users.Add(user);
            }

            return idols;
        }

        public void initCurrentUserAndIdol(string idolName, string userName)
        {
            var currentInfo = DB.SetUserAndIdol(userName, idolName, this.Idols, this.users);
            if(currentIdol != null || currentUser != null) { 
            if (currentIdol.IName != idolName || currentUser.UName != userName)
                DB.updateScoreTable(currentIdol.Inumber, currentUser.Unumber, currentScore);
                currentScore = 0;
            }

            currentIdol = currentInfo.Item1;
            currentUser = currentInfo.Item2;
            MessageBox.Show($"현재 아이돌 : {currentIdol.IName} 현재 유저 : {currentUser.UName}");
            DB.updateScoreTable(currentIdol.Inumber, currentUser.Unumber, currentScore);
            currentScore = 0;
            UpdateRank?.Invoke(this, new args.UpdateRankArgs(this.Idols, this.users));
        }

        public void ResetProgressBar()
        {
            
        }

        internal void StopCalcScore()
        {
            if(StopCalcTokenSource == null) return;
            StopCalcTokenSource.Cancel();
            this.kinectController.StopDisplayColorFrame();
        }

        ///위 메소드가 실행되면 발동
        ///

        public event EventHandler<args.UpdateRankArgs> UpdateRank;




    }
}
