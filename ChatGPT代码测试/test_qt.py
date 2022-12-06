from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QVBoxLayout
from PyQt5.QtCore import QCoreApplication

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("PyQt程序")
        self.setGeometry(400,400,300,260)
        layout = QVBoxLayout(self)
        self.layout = layout

        # 创建三个按钮
        btn1 = QPushButton("按钮1", self)
        btn2 = QPushButton("按钮2", self)
        btn3 = QPushButton("按钮3", self)

        self.layout.addWidget(btn1)
        self.layout.addWidget(btn2)
        self.layout.addWidget(btn3)

        # 设置按钮的点击事件
        btn1.clicked.connect(self.btn1_clicked)
        btn2.clicked.connect(self.btn2_clicked)
        btn3.clicked.connect(self.btn3_clicked)

    def btn1_clicked(self):
        # 关机
        QCoreApplication.quit()

    def btn2_clicked(self):
        # 弹出恭喜的提示
        QMessageBox.information(self, "恭喜", "您点击了按钮2！")

    def btn3_clicked(self):
        # 弹出恭喜的提示
        QMessageBox.information(self, "恭喜", "您点击了按钮3！")

if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.show()
    app.exec_()

