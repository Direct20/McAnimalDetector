from PyQt5 import QtCore, QtGui, QtWidgets
from ui.LoginDialogUI import Ui_LoginDialog
import cv2
import utils
import logging
import numpy as np
from ultralytics import YOLO
import sqlite3
from PyQt5.QtWidgets import QMessageBox
from MainWindow import MainWindow
from RegisterDialog import RegisterDialog
import traceback
import hashlib


class LoginDialog(QtWidgets.QDialog):

    def __init__(self):
        super(LoginDialog, self).__init__()
        self.ui = Ui_LoginDialog()
        self.ui.setupUi(self)

        self.ui.loginBtn.clicked.connect(self.onLoginBtnClicked)
        self.ui.registerBtn.clicked.connect(self.onRegisterBtnClicked)
        self.registerDialog = RegisterDialog()
        self.mainWindow = MainWindow()
        
        # utils.apply_style_sheet(self)

    def onRegisterBtnClicked(self):
        self.registerDialog.show()

    def onLoginBtnClicked(self):
        acc = self.ui.accountEdit.text()
        passwd = self.ui.passwordEdit.text()
        cnn = self.createConnection()
        if cnn:
            self.loginUser(cnn, acc, passwd)
            cnn.close()
        # else:
        # QMessageBox.warning(self,)

    def createConnection(self):
        try:
            conn = sqlite3.connect('./config/user.db')
            return conn
        except BaseException as e:
            traceback_info = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__))

            QMessageBox.critical(None, '错误',
                                 f"发生错误: {str(e)}\n\n堆栈跟踪:\n{traceback_info}",
                                 QMessageBox.StandardButton.Ok)

    # def hash_password(self, password):
    #     hash_object = hashlib.sha256(password.encode('utf-8'))
    #     return hash_object.hexdigest()

    def loginUser(self, conn, account, password):
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user WHERE username=? AND password=?",
                (account, password))
            result = cursor.fetchone()
            if result:
                QMessageBox.information(self, '信息', '登录成功！')
                self.mainWindow.show()
                self.close()
            else:
                QMessageBox.information(self, '信息', '用户名或密码错误。')
        except BaseException as e:
            traceback_info = ''.join(
                traceback.format_exception(type(e), e, e.__traceback__))

            QMessageBox.critical(None, '错误',
                                 f"发生错误: {str(e)}\n\n堆栈跟踪:\n{traceback_info}",
                                 QMessageBox.StandardButton.Ok)
