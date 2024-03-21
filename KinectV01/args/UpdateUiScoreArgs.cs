using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KinectV01.args
{
    public class UpdateUiScoreArgs
    {
        public Idol Idol { get; private set; }
        public User User { get; private set; }

        public int Score { get; private set; }

        public UpdateUiScoreArgs(Idol idol, User user, int score)
        {
            Idol = idol;
            User = user;
            Score = score;
        }
    }
}
