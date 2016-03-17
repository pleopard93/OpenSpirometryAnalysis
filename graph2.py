from Tkinter import *
import json
from scipy import signal
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from numpy import arange, sin, pi
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
import pdb

class graph2:
	def __init__(self,  frame):
		# setup for the figure
		self.f = Figure(figsize=(5, 4), dpi=100)
		self.a = self.f.add_subplot(111)

		# do some initial plotting
		t = arange(0.0, 3.0, 0.01)
		s = sin(2*pi*t)

		# add axes
		self.a.set_xlabel('time (s)')
		self.a.set_ylabel('frequency (Hz)')

		# add to tk canvas
		self.a.plot(t, s)
		self.canvas = FigureCanvasTkAgg(self.f, master=frame)
		self.canvas.show()
		self.canvas.get_tk_widget().pack(side=LEFT, fill=X, expand=1)


	def clearGraph(self):
		self.a.clear()
		self.canvas.draw()

	def showGraph(self, data, dirname = '', waveform=None):
		self.clearGraph()
		filename = data["RecordedAudioFilenameForEffort"]
		fs, audio_data = wavfile.read(dirname + filename)

		nFFT = 4096
		nOverlap = int(nFFT * 0.75)

		if waveform is not None:
			# plot the waveform of the test
			t = np.arange(0,len(waveform["Data"])).astype(np.float)/float(waveform["Header"]["Freq"])
			self.a.plot(t,waveform["Data"])
			# add proper axes
			if waveform["Header"]["Type"] =="VT":
				self.a.set_xlabel('time (s)')
				self.a.set_ylabel('volume (L)')
			elif waveform["Header"]["Type"] =="FT":
				self.a.set_xlabel('time (s)')
				self.a.set_ylabel('flow (L/sec)')

		self.canvas.show()

		# custom add graphic
		f = plt.figure(1)
		f.clear()
		plt.ion()
		ax1 = plt.subplot(211)
		t = np.arange(0,len(audio_data)).astype(np.float)/float(fs)
		ax1.plot(t,audio_data)

		ax2 = plt.subplot(212,sharex=ax1)
		Pxx, freqs, t, im = plt.specgram(audio_data, NFFT=nFFT, Fs=fs, noverlap=nOverlap)

		plt.show()

