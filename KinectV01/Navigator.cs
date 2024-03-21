using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KinectV01
{
    public class Navigator
    {
        private EnterWindow currentEnterWindow = null;
        public void showEnterWindow()
        {
            currentEnterWindow = new EnterWindow();
            currentEnterWindow.ShowDialog();
        }
        public void exitEnterWindow() 
        {
            if(currentEnterWindow == null) return;
            currentEnterWindow.Close();
        }
    }
}
