using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KinectV01.args
{
    public class UpdateRankArgs : EventArgs
    {
        public List<Idol> Idols { get; private set; }
        public List<User> Users { get; private set; }

        public UpdateRankArgs(List<Idol> idols, List<User> users)
        {
            Idols = idols;
            Users = users;
        }
    }
}
