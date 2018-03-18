import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import picamera
import cv2
from main2 import *

OFFSET_BETWEEN_WIDGETS = 10

'''
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
'''


class LineEdit(QLineEdit):
    def __init__(self, valid_range=None, decimal=None, placeholder='', contents='', parent=None):
        super().__init__(contents, parent)
        self.setPlaceholderText(placeholder)
        if valid_range is not None:
            min_range, max_range = valid_range
            if decimal is None:
                validator = QIntValidator(min_range, max_range)
            else:
                validator = QDoubleValidator(min_range, max_range, decimal)
            self.setValidator(validator)
        self.textEdited.connect(self.on_edited)

    def on_edited(self):
        received_text = self.text()
        is_valid = self.validator().validate(received_text, 0)[0]
        if is_valid == QValidator.Acceptable:
            self.setStyleSheet("background-color: rgb(255, 255, 255); color: 0, 0, 0")
        else:
            self.setStyleSheet("background-color: rgb(255, 199, 206); color: 156, 0, 6")
        '''
        if self.parent().whatsThis() == 'InputFieldWidget':
            try:
                new_value = float(received_text)
            except ValueError:
                return
            self.parent().on_edit(self, new_value)
        '''


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


class PopUp(QDialog):
    def __init__(self, title, text, yn, pos=None, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        label = QLabel(text, parent=self)
        label.move(QPoint(20, 20))
        label.adjustSize()

        button_size=QSize(60, 20)
        button_pos_y = label.pos().y() + label.height() + OFFSET_BETWEEN_WIDGETS
        if yn:
            # This popup should return a yes/no response
            button_yes = QPushButton('Yes', parent=self)
            button_yes.clicked.connect(self.yes)
            button_yes.resize(button_size)
            button_yes_pos = QPoint(label.pos().x() + (label.width() - OFFSET_BETWEEN_WIDGETS) // 2 - button_size.width(),
                                    button_pos_y)
            button_yes.move(button_yes_pos)
            button_no = QPushButton('No', parent=self)
            button_no.clicked.connect(self.no)
            button_no.resize(button_size)
            button_no_pos = QPoint(label.pos().x() + (label.width() + OFFSET_BETWEEN_WIDGETS) // 2, button_pos_y)
            button_no.move(button_no_pos)
        else:
            # This popup doesn't require a response
            button_ok = QPushButton('Ok', parent=self)
            button_ok.clicked.connect(self.done)
            button_ok.resize(button_size)
            button_ok_pos = QPoint(label.pos().x() + (label.width() - button_size.width()) // 2, button_pos_y)
            button_ok.move(button_ok_pos)

    def yes(self):
        self.done(QDialog.Accepted)

    def no(self):
        self.done(QDialog.Rejected)


class App(QDialog):
    def __init__(self, title, pos=None, parent=None):
        super().__init__(parent=parent)
        self.__widgets = {}

        self.setWindowTitle(title)
        if pos is not None:
            self.setGeometry(pos)

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

    def add_input_field(self, name, label, pos=QPoint(0, 0), size=QSize(40, 20), tooltip=None, valid_range=None,
                        decimal=None, contents=''):
        # Check if this name is already used for another widget
        if name in self.__widgets:
            print('WARNING: widget %s already exists within this app. Try different name' % name)
            return

        # Create label & input field as specified
        label_range = ''
        if valid_range is not None:
            if decimal is None:
                label_range = '(%d - %d)' % valid_range
            else:
                label_range = '(%.*f - %.*f)' % (decimal, valid_range[0], decimal, valid_range[1])
        text = QLabel(label + ' ' + label_range, self)
        text.adjustSize()
        if tooltip is not None:
            text.setToolTip(tooltip)
        field = LineEdit(valid_range=valid_range, decimal=decimal, contents=contents, parent=self)
        field.resize(size)

        text.move(pos)
        field_pos = QPoint(pos.x() + text.size().width() + OFFSET_BETWEEN_WIDGETS, pos.y())
        field.move(field_pos)

        # Attach widgets to the app
        self.__widgets[name+'_TEXT'] = text
        self.__widgets[name+'_FIELD'] = field

    def get_widget(self, name):
        return self.__widgets[name]

    def popup(self, text, yn=False):
        popup = PopUp('Warning', text, yn)
        return popup.exec()


class MainApp(App):
    def __init__(self, title, dict_settings=None, pos=None, parent=None):
        super().__init__(title, pos=pos, parent=parent)
        self.__calib_offset = 0
        self.__warp_matrix = None

        if dict_settings is not None:
            self.__dict_settings = dict_settings
        else:
            self.__dict_settings = {'RC_NUM_FRAMES': 10,
                                    'RC_RADIUS': 25,
                                    'MAX_FPS': 20,
                                    'MAX_THREAD': 6,
                                    'BLUR_SIZE': 5,
                                    'SD_BRIGHT_THRESHOLD': 40,
                                    'CD_BRIGHT_THRESHOLD': 30,
                                    'CD_BRIGHT_PERCENTILE': 50,
                                    'CD_COORD_WEIGHT': 0.5,
                                    }

        self.add_push_button(name='RECALIBRATE', label='Calibrate', on_click=self.calibrate,
                             pos=QPoint(20, 20), size=QSize(100, 30))
        self.add_push_button(name='ACTIVATE', label='Start', on_click=self.activate,
                             pos=QPoint(20, 60), size=QSize(100, 30))
        self.add_push_button(name='OPEN_SETTINGS', label='Settings', on_click=self.show_settings,
                             pos=QPoint(20, 100), size=QSize(100, 30))

        # Initialize picamera
        self.__camera = picamera.PiCamera(sensor_mode=5)
        self.__camera.color_effects = (128, 128)  # Grayscale
    
    def show_settings(self):
        dialog_setting = SettingsApp('Settings', self.__dict_settings, parent=self)
        ret = dialog_setting.exec()
        if ret == QDialog.Accepted:
            self.__dict_settings = dialog_setting.dict_settings

    def calibrate(self):
        # TODO: grab screen resolution
        screen_width = 800
        screen_height = 600

        # TODO: integrate calibration implementation
        offset = self.__calib_offset
        calib_coords = calibrate(self.__camera, offset=offset)
        if calib_coords is not None:
            # Calculate transformation matrix to convert from calibration coordinates to screen coordinates
            np_calib_coords = np.float32([
                [calib_coords[0][0], calib_coords[0][1]],
                [calib_coords[1][0], calib_coords[1][1]],
                [calib_coords[2][0], calib_coords[2][1]],
                [calib_coords[3][0], calib_coords[3][1]]
            ])
            np_screen_coords = np.float32([
                [offset, offset],                                   # Top Left
                [screen_width - offset, offset],                    # Top Right
                [screen_width - offset, screen_height - offset],    # Bottom Right
                [offset, screen_height - offset]                    # Bottom Left
            ])
            self.__warp_matrix = cv2.getPerspectiveTransform(np_calib_coords, np_screen_coords)
            self.get_widget('RECALIBRATE').setText('Recalibrate')
            print(self.__warp_matrix)
        else:
            self.popup('Calibration failed. Please try again.')


    def activate(self):      
        if self.__warp_matrix is None:
            yn = self.popup('The camera hasn\'t been calibrated yet. Would you like to calibrate now?', yn=True)
            if yn == QDialog.Accepted:
                self.calibrate()
            else:
                return
        # TODO: integrate processing framework
        process_output = ProcessOutput()
        consumer = Consumer()
        dr.show_clear()	
        cam_thread = CameraThread(self.__camera_pi, process_output)
        pass


class SettingsApp(App):
    def __init__(self, title, dict_settings, pos=None, parent=None):
        super().__init__(title, pos, parent)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.dict_settings = dict_settings
        # Parameters for right click register
        self.add_input_field(name='RC_NUM_FRAMES', label='Duration of frames for right click',
                             pos=QPoint(20, 20), valid_range=(3, 20), 
                             contents=str(self.dict_settings['RC_NUM_FRAMES']),
                             tooltip='Holding the pointer for this many frames will register as right click')
        self.add_input_field(name='RC_RADIUS', label='Radius in pixels for right click',
                             pos=QPoint(20, 50), valid_range=(5, 50), contents=str(self.dict_settings['RC_RADIUS']),
                             tooltip='Holding the pointer within this radius will register as right click')
        # Parameters for performance restriction
        self.add_input_field(name='MAX_FPS', label='Maximum fps to process', pos=QPoint(20, 100),
                             valid_range=(15, 30), contents=str(self.dict_settings['MAX_FPS']),
                             tooltip='Lower fps may result in less responsive behaviour, '
                                     'but will limit required processing power')
        self.add_input_field(name='MAX_THREAD', label='Number of concurrent threads', pos=QPoint(20, 130),
                             valid_range=(1, 12), contents=str(self.dict_settings['MAX_THREAD']),
                             tooltip='Less threads may result in less responsive behaviour, '
                                     'but will limit required processing power')
        # Parameters for core detection algorithm
        self.add_input_field(name='BLUR_SIZE', label='Mask size of Gaussian blur', pos=QPoint(20, 180),
                             valid_range=(3, 11), contents=str(self.dict_settings['BLUR_SIZE']))
        self.add_input_field(name='SD_BRIGHT_THRESHOLD',
                             label='Brightness threshold for source detection',
                             pos=QPoint(20, 210), valid_range=(20, 60),
                             contents=str(self.dict_settings['SD_BRIGHT_THRESHOLD']))
        self.add_input_field(name='CD_BRIGHT_THRESHOLD',
                             label='Brightness threshold for coordinate detection',
                             pos=QPoint(20, 240), valid_range=(20, 60),
                             contents=str(self.dict_settings['CD_BRIGHT_THRESHOLD']))
        self.add_input_field(name='CD_BRIGHT_PERCENTILE',
                             label='Brightness percentile for coordinate detection',
                             pos=QPoint(20, 270), valid_range=(20, 100),
                             contents=str(self.dict_settings['CD_BRIGHT_PERCENTILE']))
        self.add_input_field(name='CD_COORD_WEIGHT',
                             label='Weight of brightest point to compute projection coordinate',
                             pos=QPoint(20, 300), valid_range=(0.0, 1.0), decimal=2,
                             contents=str(self.dict_settings['CD_COORD_WEIGHT']))
        self.add_push_button(name='EXIT_SAVE', label='Save', on_click=self.save_on_exit, pos=QPoint(200, 350))
        self.add_push_button(name='EXIT_CANCEL', label='Cancel', on_click=self.cancel_on_exit, pos=QPoint(270, 350))

    def save_on_exit(self):
        # Save each parameter's value into dict - retrieved by main app
        for param_name in self.dict_settings.keys():
            param_widget = self.get_widget(param_name + '_FIELD')
            is_valid, param_text, _ = param_widget.validator().validate(param_widget.text(), 0)
            if is_valid == QValidator.Acceptable:
                try:
                    param_value = int(param_text)
                except ValueError:
                    param_value = float(param_text)
                self.dict_settings[param_name] = param_value
            else:
                self.popup('Some of the parameters\' value does not fit within the specified range.'
                           '\nPlease check the values again.',
                           yn=False)
                return
        self.done(QDialog.Accepted)

    def cancel_on_exit(self):
        self.done(QDialog.Rejected)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_main = MainApp('LhiteBoard')
    app_main.show()
    sys.exit(app.exec_())
