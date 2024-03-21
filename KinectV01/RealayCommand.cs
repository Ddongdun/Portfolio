using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace KinectV01
{
    public class RelayCommand : ICommand
    {
        private readonly Action<object> _executeWithParameter;
        private readonly Action _executeWithoutParameter;
        private readonly Func<object, bool> _canExecute;

        // 매개변수가 있는 생성자
        public RelayCommand(Action<object> execute, Func<object, bool> canExecute = null)
        {
            _executeWithParameter = execute ?? throw new ArgumentNullException(nameof(execute));
            _canExecute = canExecute;
        }

        // 매개변수가 없는 생성자
        public RelayCommand(Action execute, Func<bool> canExecute = null)
        {
            _executeWithoutParameter = execute ?? throw new ArgumentNullException(nameof(execute));
            _canExecute = canExecute != null ? new Func<object, bool>(x => canExecute()) : null;
        }

        public bool CanExecute(object parameter)
        {
            return _canExecute == null || _canExecute(parameter);
        }

        public void Execute(object parameter)
        {
            if (_executeWithParameter != null)
                _executeWithParameter(parameter);
            else
                _executeWithoutParameter?.Invoke();
        }

        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }
    }


}
