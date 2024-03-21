using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KinectV01.args
{
    public class UpdateURLArgs : EventArgs
    {
        public string URL { get; private set; }

        public UpdateURLArgs(String url)
        {
            URL = url;
        }
    }
}
