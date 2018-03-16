import sys
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


@pyqtSlot()
def on_click_b1(self):
    print("B1 clicked")


@pyqtSlot()
def on_click_b2(self):
    print("B2 clicked")


class InputFieldWidget(QWidget):
    def __init__(self, label, min_value, max_value, default, interval, parent=None, pos=QPoint(0, 0)):
        super().__init__(parent=parent)
        self._label = QLabel(label, parent=self)
        self._label.resize(len(label) * 8, 20)
        self._line_edit = LineEdit(contents=str(default), parent=self)
        self._line_edit.resize(30, 20)
        self._slider = Slider(min_value, max_value, default, interval, parent=self)
        self._slider.resize(100, 20)
        self._value = None
        self.setWhatsThis('InputFieldWidget')
        self.move(pos)

    def move(self, pos):
        offset = 5
        line_edit_pos = QPoint(pos.x() + self._label.size().width() + offset, pos.y())
        slider_pos = QPoint(line_edit_pos.x() + self._line_edit.size().width() + offset, pos.y())
        self._label.move(pos)
        self._line_edit.move(line_edit_pos)
        self._slider.move(slider_pos)
        pass

    def on_edit(self, edited_widget, value):
        self._value = value
        if edited_widget is self._line_edit:
            print('line_edit: %f' % self._value)
            scaled_value = self._slider.scale(self._value)
            self._slider.setValue(scaled_value)
        elif edited_widget is self._slider:
            print('slider: %f' % self._value)
            self._line_edit.setText(str(self._value))


class LineEdit(QLineEdit):
    def __init__(self, placeholder='', contents='', parent=None):
        super().__init__(contents, parent)
        self.setPlaceholderText(placeholder)
        # TODO: add validator
        self.textEdited.connect(self.on_edited)

    def on_edited(self):
        if self.parent().whatsThis() == 'InputFieldWidget':
            received_text = self.text()
            try:
                new_value = float(received_text)
            except ValueError:
                return
            self.parent().on_edit(self, new_value)


class Slider(QSlider):
    def __init__(self, min_value, max_value, default, interval, parent=None):
        super().__init__(Qt.Horizontal, parent)
        # Since QSlider can only display integer value, the the range must be scaled
        self._min = min_value
        self._max = max_value
        self._interval = interval
        scaled_default = self.scale(default)

        # The slider will have a range of [0, num_interval] instead of [min_value, max_value]
        print((max_value - min_value) / interval)
        self.setRange(0, (max_value - min_value) / interval)
        self.setValue(scaled_default)

        self.setTickPosition(QSlider.TicksBelow)
        self.setTickInterval(interval)
        self.sliderMoved.connect(self.on_slider_moved)

    def scale(self, value):
        return (value - self._min) / self._interval

    def descale(self, scaled_value):
        return self._min + self._interval * scaled_value

    def on_slider_moved(self):
        if self.parent().whatsThis() == 'InputFieldWidget':
            scaled_value = self.value()
            print(scaled_value)
            new_value = self.descale(scaled_value)
            self.parent().on_edit(self, new_value)


class App(QWidget):
    def __init__(self, title, rect):
        super().__init__()
        self.__widgets = {}

        self.setWindowTitle(title)
        self.setGeometry(rect)

    def add_push_button(self, name, label, on_click, pos=QPoint(0, 0), size=QSize(60, 20), tooltip=None):
        # Check if this name is already used for another widget
        if name in self.__widgets:
            print('WARNING: widget %s already exists within this app. Try different name' % name)
            return

        # Create button as specified
        button = QPushButton(label, self)
        if tooltip is not None:
            button.setToolTip(tooltip)
        button.move(pos)
        button.resize(size)
        button.clicked.connect(on_click)

        # Attach the widget to the app
        self.__widgets[name] = button

    def add_input_field(self, name, label, pos=QPoint(0, 0), size=QSize(60, 20), tooltip=None):
        # Check if this name is already used for another widget
        if name in self.__widgets:
            print('WARNING: widget %s already exists within this app. Try different name' % name)
            return

        # Create label & input field as specified
        text = QLabel(label, self)
        text.move(pos)
        if tooltip is not None:
            text.setToolTip(tooltip)
        field = LineEdit(parent=self)
        field_pos = QPoint(pos.x() + len(label) * 8, pos.y())
        field.move(field_pos)

        # Attach widgets to the app
        self.__widgets[name+'_text'] = text
        self.__widgets[name+'_field'] = field

    def add_slider(self, name, label, min_value, max_value, default, pos=QPoint(0, 0), size=QSize(60, 20), tooltip=None):
        # Check if this name is already used for another widget
        if name in self.__widgets:
            print('WARNING: widget %s already exists within this app. Try different name' % name)
            return

        # Create a slider as specified
        text = QLabel(label, self)
        text.move(pos)
        if tooltip is not None:
            text.setToolTip(tooltip)

        slider_pos = QPoint(pos.x() + len(label) * 9, pos.y())
        slider = Slider(min_value, max_value, default, parent=self)
        slider.move(slider_pos)

        # Attach the widget to the app
        self.__widgets[name+'_text'] = text
        self.__widgets[name+'_slider'] = slider

    def get_widget(self, name):
        return self.__widgets[name]

    def refresh_app(self):
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    start_x = 0
    start_y = 0
    width = 200
    height = 200
    app_settings = App('Settings', QRect(start_x, start_y, width, height))
    #app_settings.add_push_button('RECALIB', 'Recalibrate', on_click_b1, QPoint(20, 20), QSize(100, 30), 'Hello World!')
    #app_settings.add_push_button('PAUSE', 'Pause', on_click_b2, QPoint(20, 50), QSize(100, 30), 'Goodbye World!')
    #app_settings.add_input_field('PARAM_1', 'param1', QPoint(20, 80), QSize(100, 30), 'Hello World!')
    #app_settings.add_input_field('PARAM_2', 'param2', QPoint(20, 110), QSize(100, 30), 'Goodbye World!')
    #app_settings.add_slider('SLIDER_1', '# of frames', 10, 30, 20, QPoint(20, 140), QSize(100, 30), 'Hello World!')
    InputFieldWidget(label='Param 1', min_value=10, max_value=30, default=20, interval=5, parent=app_settings)

    app_settings.refresh_app()
    sys.exit(app.exec_())
