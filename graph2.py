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

# from skimage import morphology as mp

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

		print "Processing Data ...",

		# custom add graphic
		f = plt.figure(1)
		f.clear()
		plt.ion()

		ax1 = plt.subplot(211)
		t = np.arange(0,len(audio_data)).astype(np.float)/float(fs)
		ax1.plot(t,audio_data)



		# plot spectrogram and alsget handle to it

		ax2 = plt.subplot(2,1,2,sharex=ax1)
		P_skip = 128
		P_NFFT = 2048*2
		P,P_f,P_t,im = plt.specgram(audio_data, NFFT=P_NFFT, Fs=fs, noverlap=P_NFFT-P_skip)
		P = P.astype(np.float)
		ax2.pcolorfast(P_t,P_f,20*np.log(P),cmap=plt.cm.bone)

		ax2.pcolorfast(P_t,P_f,20*np.log(P),cmap=plt.cm.bone)

		plt.show()
		return

		# Find the section of the signal we're interested in by just looking at the energy. Yay Parseval's Theorem!
		energy = np.sum(P, axis=0)
		energy = energy*(fs/2)/np.max(energy)
		#plt.plot(P_t, energy, 'k')
		PEF_index = np.argmax(energy)

		# get a normalized from 0-255 version of the spectrogram
		P_normalized = np.log(P)
		P_normalized = (P_normalized - np.min(P_normalized)) / (np.max(P_normalized)-np.min(P_normalized))*255
		P_normalized_uint = P_normalized.astype(np.uint8) # make integer for easy comparison

		# dilate on the columns to get local maxima
		P_dilate = mp.dilation(P_normalized_uint, np.matrix(np.ones((40,1))).astype(np.uint8))

		# P_local_max = mp.binary_dilation(P_dilate==P_normalized,mp.disk(1)) & (P_normalized>190)

		# get a mask of where the energy is large for resonance finding
		mask = np.repeat((energy>np.mean(energy)*0.1),P.shape[0], axis=0).reshape(P.T.shape).T
		mask = mp.binary_closing(mask,np.ones((1,50)))
		mask = mp.binary_erosion(mask,np.ones((1,50)))

		# find local maxima along columns that also have good magnitude 
		P_local_max = (P_dilate==P_normalized_uint) & (P_normalized>170) & mask

		# now use closing to try and connect pixels along an 'X'
		P_local_max = mp.binary_closing(P_local_max,np.eye(3))
		P_local_max = mp.binary_closing(P_local_max,np.eye(3).T)
		#P_local_max = mp.skeletonize(P_local_max)

		# remove anything that is not a certain pixel area
		P_local_max = P_local_max.astype(np.bool)
		P_local_max = mp.remove_small_objects(P_local_max, min_size = 50, connectivity=13, in_place=True)

		# plot the output
		ax3 = plt.subplot(313,sharex=ax1, sharey=ax2)
		Ptmp = P_normalized.copy()
		Ptmp[P_local_max==True] = 0
		ax3.pcolorfast(P_t, P_f, Ptmp, cmap=plt.cm.bone)

		plt.show()

		print "Done!"

