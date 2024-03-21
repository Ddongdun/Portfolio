using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FirstPage
{

    public class IdolTotal
    {
        private static Idol instance;
        private string iName;
        private int iScore;
        private List<String> idolUsers = new List<string>();
        private int number;
        private List<String> usersscore = new List<string>();

        
        //아이돌 명
        public string IName
        {
            get { return iName; }
            set { iName = value; }
        }

        // 누적된 총 점수를 담을 변수
        public int IScore
        {
            get { return iScore; }
            set { iScore = value; }
        }

        public int Number
        {
            get { return number; }
            set { number = value; }
        }

        public List<String> IdolUsers
        {
            get { return idolUsers; }
            set { idolUsers = value; }
        }

        public List<String> UsersScore
        {
            get { return usersscore; }
            set { usersscore = value; }
        }

        // 아이돌이 저장하고 있는 각 유저별 누적 응원 점수
        // 파일에 저장 할 때 유저명:점수 유저명: 점수로 저장

    }
 }
          
