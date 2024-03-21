using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KinectV01
{
    public class User
    {
        private string uName;
        private int uScore;
        private int unumber;

        private Dictionary<String, int> Useridols = new Dictionary<string, int>();

        //아이돌 명
        public string UName
        {
            get { return uName; }
            set { uName = value; }
        }

        // 누적된 총 점수를 담을 변수
        public int UScore
        {
            get { return uScore; }
            set { uScore = value; }
        }

        public int Unumber
        {
            get { return unumber; }
            set { unumber = value; }
        }

        // 아이돌이 저장하고 있는 각 유저별 누적 응원 점수
        // 파일에 저장 할 때 유저명:점수 유저명: 점수로 저장
        public Dictionary<String, int> UserIdols
        {
            get { return Useridols; }
            set { Useridols = value; }
        }

    }
}