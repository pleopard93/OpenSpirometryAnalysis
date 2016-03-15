from Tkinter import *
import json
from scipy import signal
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from numpy import arange, sin, pi

class graph2:
	def __init__(self,  frame):
		f = Figure(figsize=(5, 4), dpi=100)
		self.a = f.add_subplot(111)
		t = arange(0.0, 3.0, 0.01)
		s = sin(2*pi*t)

		self.a.plot(t, s)
		self.canvas = FigureCanvasTkAgg(f, master=frame)
		self.canvas.show()
		self.canvas.get_tk_widget().pack(side=LEFT, fill=X, expand=1)

	def clearGraph(self):
		self.a.clear()
		self.canvas.draw()

	def showGraph(self, data):
		self.clearGraph()
		# f, Pxx_den = signal.periodogram(data["FlowCurveInLitersPerSecond"])

		# plt.semilogy(f, Pxx_den)
		# plt.xlabel('frequency [Hz]')
		# plt.ylabel('PSD [V**2/Hz]')
		# plt.show()
