
import pandas as pd
import sys
from PySide2.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider
)
from PySide2.QtCore import Qt
from PySide2 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide2.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QApplication, QGridLayout, QPushButton
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Draw on Image https://stackoverflow.com/questions/51475306/drawing-on-top-of-image-in-pyqt5-tracing-the-mouse
# Drop Down and than mark related Parts in figure?  
class MplCanvas(FigureCanvasQTAgg):
    '''MatPlotLib '''

    def __init__(self,parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MplTimeSeriesCanvas(FigureCanvasQTAgg):
    '''MatPlotLib '''

    def __init__(self,parent=None, i=None, exp=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axs = fig.add_subplot(111)
        axn012 =self.axs.twinx()
        sns.heatmap(
            exp.reshape(1, -1),
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=self.axs,
                vmin=0,
                vmax=1,
            )
        sns.lineplot(
                x=range(0, len(i.reshape(-1))),
                y=i.flatten(),
                ax=axn012,
                color="white",
            )
        super(MplTimeSeriesCanvas, self).__init__(fig)


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])


class MainWindow(QMainWindow):

    def __init__(self, input,label, exp, data_type='img'):
        '''
        Attributes
            input np.array: input in already correct shape 
        '''
        super(MainWindow, self).__init__()
        self.data='Starting Data'
        self.label=label
        self.feedback=(None,None)
        if data_type == 'img':
            sc = MplCanvas(self, width=5, height=4, dpi=100)
            sc.axes.imshow(input)
            toolbar = NavigationToolbar(sc, self)
        elif data_type == 'timeseries':
            
            sc = MplTimeSeriesCanvas(self, input,exp, width=5, height=4, dpi=100)

            toolbar = NavigationToolbar(sc, self)


        layout_outer= QGridLayout()
        button_layout = QHBoxLayout()
        layout_inner_right = QVBoxLayout()
        layout_checkboxes= QGridLayout()
        layout_outer.addLayout(layout_inner_right,0,1)
        layout_outer.addWidget(toolbar)
        layout_outer.addWidget(sc,0,0)
        
        # as 'Heatmap'
        #sc_explanation = MplCanvas(self, width=3, height=3, dpi=100)
        #sc_explanation.axes.imshow(exp)
        #layout_inner_right.addWidget(sc_explanation)
        
        # as 'Plot'
        sc_explanation_plot = MplCanvas(self, width=3, height=3, dpi=100)
        sc_explanation_plot.axes.bar(range(0,len(exp.reshape(-1))),exp.reshape(-1))
        layout_inner_right.addWidget(sc_explanation_plot)
        toolbar2 = NavigationToolbar(sc, sc_explanation_plot)



        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout_outer)
        self.setCentralWidget(widget)

        self.show()

        #CheckBoxes if necessary
        layout_inner_right.addLayout(layout_checkboxes)
        res_check= np.zeros_like(exp)
        check=[]
        #print(len(exp))
        for i in range(0,len(exp.reshape(-1))):
            #print(i)
            c= QCheckBox(f"{i}", self)
            c.stateChanged.connect(self.checkBoxChange)
            #print(i%5)
            #print(int(i/5))
            layout_checkboxes.addWidget(c,i%5,int(i/5))
            check.append(c)
        self.check = check 

        #self.setWindowTitle("My App")
        button_overlay = QPushButton("Overlay")
        button_overlay.setCheckable(True)
        button_overlay.clicked.connect(self.the_button_was_clicked)
        layout_inner_right.addWidget(button_overlay)

        #self.setWindowTitle("My App")

        widget_label = QLineEdit()
        widget_label.setMaxLength(10)
        widget_label.setPlaceholderText(f"Predicted Label: {label}")
        self.widget_label=widget_label
        layout_inner_right.addWidget(widget_label)
        #TODO For Tabular Data, see https://www.pythonguis.com/tutorials/pyside-qtableview-modelviews-numpy-pandas/
        #self.table = QtWidgets.QTableView()
         #self.setWindowTitle("My App")
        button_Finished = QPushButton("Finish")
        button_Finished.clicked.connect(self.finished)
        layout_inner_right.addWidget(button_Finished)

    def text_changed(self, s):
        self.label=s

    def text_edited(self, s):
        self.label=s

    def checkBoxChange(self):
        pass

    def the_button_was_clicked(self):
        print("Clicked!")
    
    def finished(self):
        self.data='Ending Data'#
        feat=np.array([a.isChecked() for a in self.check]).astype(int)
        self.feedback=(self.widget_label.text(),feat)
        self.data=self.feedback
        self.close()
        return self.feedback
