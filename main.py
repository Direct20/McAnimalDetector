import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from LoginDialog import LoginDialog
from MainWindow import MainWindow
import traceback
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
from qt_material import apply_stylesheet


def main():
    exit_code = 0
    try:
        QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QtGui.QGuiApplication.setAttribute(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QtWidgets.QApplication(sys.argv)
        
        apply_stylesheet(app, theme='light_blue.xml')
        
        entry_window = LoginDialog()
        entry_window.show()
        exit_code = app.exec_()
    except BaseException as e:
        traceback_info = ''.join(
            traceback.format_exception(type(e), e, e.__traceback__))

        QMessageBox.critical(None, '错误',
                             f"发生错误: {str(e)}\n\n堆栈跟踪:\n{traceback_info}",
                             QMessageBox.StandardButton.Ok)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
