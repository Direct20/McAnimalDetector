from PyQt5 import QtCore, QtGui, QtWidgets
from ui.RegisterDialogUI import Ui_RegisterDialog
import cv2
import utils
import logging
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import QMessageBox
import sqlite3
import hashlib
import traceback


class RegisterDialog(QtWidgets.QDialog):

    def __init__(self):
        super(RegisterDialog, self).__init__()
        self.ui = Ui_RegisterDialog()
        self.ui.setupUi(self)

        self.ui.registerBtn.clicked.connect(self.onRegisterBtnClicked)
        
        # utils.apply_style_sheet(self)

    def onRegisterBtnClicked(self):
        acc = self.ui.accountEdit.text()
        passwd = self.ui.passwordEdit.text()
        passwd2 = self.ui.passwordConfirmEdit.text()
        if acc == '':
            QMessageBox.warning(self,
                                '警告',
                                '用户名不能为空',
                                buttons=QMessageBox.StandardButton.Ok)
            return
        if passwd == '' or passwd2 == '':
            QMessageBox.warning(self,
                                '警告',
                                '密码不能为空',
                                buttons=QMessageBox.StandardButton.Ok)
            return
        if passwd != passwd2:
            QMessageBox.warning(self,
                                '警告',
                                '两次输入的密码不一致！',
                                buttons=QMessageBox.StandardButton.Ok)
            return
        cnn = self.createConnection()
        if cnn:
            self.registerUser(cnn, acc, passwd)
            cnn.close()

    def createConnection(self):
        try:
            conn = sqlite3.connect('./config/user.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS user
                            (username TEXT PRIMARY KEY, passwordTEXT)''')
            conn.commit()
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

    def registerUser(self, conn, account, password):
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO user (username, password) VALUES (?,?)",
                (account, password))
            conn.commit()
            QMessageBox.information(self, '信息', '注册成功！')
        except sqlite3.IntegrityError:
            QMessageBox.information(self, '信息', '用户名已存在，请更换用户名重新注册。')
