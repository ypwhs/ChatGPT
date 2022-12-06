import random
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.button = QPushButton('Button', self)
        self.button.clicked.connect(self.on_click)

    def on_click(self):
        if random.random() < 0.5:
            self.button.move(random.randint(0, self.width() - self.button.width()),
                             random.randint(0, self.height() - self.button.height()))
        else:
            self.button.setText(str(random.randint(0, 100)))


if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.resize(300, 300)
    window.show()
    app.exec_()
