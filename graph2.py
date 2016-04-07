from Tkinter import *
import json
from scipy import signal
from scipy.signal import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from numpy import arange, sin, pi
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
import pdb
import pandas as pd

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

	def showGraph(self, data, dirname = '', waveform=None, meta=None):
		self.clearGraph()
		filename = data["RecordedAudioFilenameForEffort"]
		fs, audio_data = wavfile.read(dirname + filename)
		

		if waveform is not None:
			# plot the waveform of the test
			t = np.arange(0,len(waveform["Data"])).astype(np.float)/float(waveform["Header"]["Freq"])
			fs_w = float(waveform["Header"]["Freq"])
			
			w = np.array(waveform["Data"]).astype(np.float)
			# add proper axes
			if waveform["Header"]["Type"] =="VT":
				
				self.a.set_xlabel('time (s)')
				self.a.set_ylabel('flow (L/sec)')
				w = (w[1:]-w[:-1])*float(waveform["Header"]["Freq"])
				w = np.hstack(([0],w))
				f = w

			elif waveform["Header"]["Type"] =="FT":
				f = w
				w = np.cumsum(w/float(waveform["Header"]["Freq"]))
				self.a.set_xlabel('time (s)')
				self.a.set_ylabel('volume (L)')
			self.a.plot(t,w)

		self.canvas.show()

		print "Processing Data ...",
		plt.style.use('ggplot')

		# custom add graphic
		f1 = plt.figure(1)
		f1.clear()
		plt.ion()

		ax1 = plt.subplot(211)
		t = np.arange(0,len(audio_data)).astype(np.float)/float(fs)
		ax1.plot(t,audio_data)
		plt.title(filename)



		# plot spectrogram and alsget handle to it

		ax2 = plt.subplot(2,1,2,sharex=ax1)
		P_skip = 128
		P_NFFT = 2048
		P,P_f,P_t,_ = plt.specgram(audio_data, NFFT=P_NFFT, Fs=fs, noverlap=P_NFFT-P_skip,window=np.hamming(P_NFFT))
		P = P.astype(np.float)
		ax2.pcolorfast(P_t,P_f,20*np.log(P),cmap=plt.cm.bone)

		ax2.pcolorfast(P_t,P_f,20*np.log(P),cmap=plt.cm.bone)

		plt.show()

		def follow_peak_in_band(band,time_energy,cooling,b_index):
		    
		    band = (band - np.min(band)) / (np.max(band)-np.min(band))*255
		    
		    b_index = np.argmax(time_energy)
		    strt_idx = np.argmax(band[:,b_index])

		    b_local_max = np.zeros(band.shape).astype(np.bool)
		    x = [strt_idx,b_index]
		    b_local_max[x[0],x[1]] = 1
		    wind_down = 4
		    wind_up = 4
		    traj = np.nan
		    cooling_iter = 1
		    for i in range(x[1],len(time_energy)):
		        freqs = band[x[0]-wind_down:x[0]+wind_up,i]
		        if len(freqs)>0:
		            j = np.argmax(freqs)
		            
		            # get possible new point
		            new_point = x[0]-wind_down+j
		            # get trajectory of last move
		            tmp_traj = float(x[0])-new_point
		            
		            if np.isnan(traj):
		                traj = tmp_traj
		            # average trajectory with new
		            traj = (cooling*tmp_traj+(1-cooling)*traj)
		            new_point = int(x[0]-traj)
		            
		            b_local_max[new_point,i]=1
		            
		            x[0] = new_point

		        
		    x = [strt_idx,b_index]
		    traj = np.nan
		    for i in range(x[1],0,-1):
		        freqs = band[x[0]-wind_down:x[0]+wind_up,i]
		        if len(freqs)>0:
		            j = np.argmax(freqs)
		            
		            # get possible new point
		            new_point = x[0]-wind_down+j
		            # get trajectory of last move
		            tmp_traj = float(x[0])-new_point
		            if np.isnan(traj):
		                traj = tmp_traj
		            # average trajectory with new
		            traj = (cooling*tmp_traj+(1-cooling)*traj)
		            new_point = int(x[0]-traj)
		            
		            b_local_max[new_point,i]=1
		            
		            x[0] = new_point
		    all_arg_max = np.zeros(time_energy.shape)
		    for i in range(all_arg_max.shape[0]):
		        all_arg_max[i] = np.argmax(b_local_max[:,i])
		    
		    return b_local_max, all_arg_max

		def inward_peak_in_band(band,time_energy,cooling,strtpt):

			band = (band - np.min(band)) / (np.max(band)-np.min(band))*255

			#b_index = np.argmax(time_energy)
			time_energy_first = np.sum(band[:,:strtpt],axis=0)

			b_index = np.argmax(time_energy_first)
			strt_idx = np.argmax(band[:,b_index])

			b_local_max = np.zeros(band.shape).astype(np.bool)
			x = [strt_idx,b_index]
			b_local_max[x[0],x[1]] = 1
			wind_down = 4
			wind_up = 2
			traj = np.nan
			cooling_iter = 1
			for i in range(x[1],len(time_energy)):
			    freqs = band[x[0]-wind_down:x[0]+wind_up,i]
			    if len(freqs)>0:
			        j = np.argmax(freqs)
			        
			        # get possible new point
			        new_point = x[0]-wind_down+j
			        # get trajectory of last move
			        tmp_traj = float(x[0])-new_point
			        
			        if np.isnan(traj):
			            traj = tmp_traj
			        # average trajectory with new
			        traj = (cooling*tmp_traj+(1-cooling)*traj)
			        new_point = int(x[0]-traj)
			        
			        b_local_max[new_point,i]=1
			        
			        x[0] = new_point

			    
			x = [strt_idx,b_index]
			traj = np.nan
			for i in range(x[1],0,-1):
			    freqs = band[x[0]-wind_down:x[0]+wind_up,i]
			    if len(freqs)>0:
			        j = np.argmax(freqs)
			        
			        # get possible new point
			        new_point = x[0]-wind_down+j
			        # get trajectory of last move
			        tmp_traj = float(x[0])-new_point
			        if np.isnan(traj):
			            traj = tmp_traj
			        # average trajectory with new
			        traj = (cooling*tmp_traj+(1-cooling)*traj)
			        new_point = int(x[0]-traj)
			        
			        b_local_max[new_point,i]=1
			        
			        x[0] = new_point
			all_arg_max = np.zeros(time_energy.shape)
			for i in range(all_arg_max.shape[0]):
			    all_arg_max[i] = np.argmax(b_local_max[:,i])

			return b_local_max, all_arg_max

		def outward_peak_in_band(band,time_energy,cooling,strtpt):

			band = (band - np.min(band)) / (np.max(band)-np.min(band))*255

			#b_index = np.argmax(time_energy)+120

			time_energy_first = np.sum(band[:,strtpt:],axis=0)
			b_index = np.argmax(time_energy_first)+strtpt
			strt_idx = np.argmax(band[:,b_index])

			b_local_max = np.zeros(band.shape).astype(np.bool)
			x = [strt_idx,b_index]
			b_local_max[x[0],x[1]] = 1
			wind_down = 4
			wind_up = 2
			traj = np.nan
			cooling_iter = 1
			for i in range(x[1],len(time_energy)):
			    freqs = band[x[0]-wind_down:x[0]+wind_up,i]
			    if len(freqs)>0:
			        j = np.argmax(freqs)
			        
			        # get possible new point
			        new_point = x[0]-wind_down+j
			        # get trajectory of last move
			        tmp_traj = float(x[0])-new_point
			        
			        if np.isnan(traj):
			            traj = tmp_traj
			        # average trajectory with new
			        traj = (cooling*tmp_traj+(1-cooling)*traj)
			        new_point = int(x[0]-traj)
			        
			        b_local_max[new_point,i]=1
			        
			        x[0] = new_point

			    
			x = [strt_idx,b_index]
			traj = np.nan
			for i in range(x[1],0,-1):
			    freqs = band[x[0]-wind_down:x[0]+wind_up,i]
			    if len(freqs)>0:
			        j = np.argmax(freqs)
			        
			        # get possible new point
			        new_point = x[0]-wind_down+j
			        # get trajectory of last move
			        tmp_traj = float(x[0])-new_point
			        if np.isnan(traj):
			            traj = tmp_traj
			        # average trajectory with new
			        traj = (cooling*tmp_traj+(1-cooling)*traj)
			        new_point = int(x[0]-traj)
			        
			        b_local_max[new_point,i]=1
			        
			        x[0] = new_point
			all_arg_max = np.zeros(time_energy.shape)
			for i in range(all_arg_max.shape[0]):
			    all_arg_max[i] = np.argmax(b_local_max[:,i])

			return b_local_max, all_arg_max

		   # custom add graphic
		f1 = plt.figure(2)
		f1.clear()
		plt.ion()

		min_f_a = 5000
		max_f_a = 5500
		energy = np.sum(P, axis=0)
		energy = energy*(fs/2)/np.max(energy)
		energy_log = np.sum(np.log(P), axis=0)
		energy_log = energy_log-np.min(energy_log)
		energy_log = energy_log*(fs/2)/np.max(energy_log)

		# likely place that PEF occurs
		P_max = self.decreasing_windowed_fft_analysis(P[:70,:], fs=fs)*70/P.shape[0]
		b, a = butter(3, .1, btype='low')
		P_max = filtfilt(b, a, P_max)

		tpeak_idx = np.argmax(energy)
		tpeak_left = max((tpeak_idx-80,0))
		tpeak_right = min((tpeak_idx+80,P.shape[1]))
		PEF_index = np.argmax(P_max[tpeak_left:tpeak_right]) + tpeak_left

		fs_effective = fs/P_skip
		time_before_PEF = fs_effective/2 
		time_after = fs_effective*3

		energy = energy[PEF_index-time_before_PEF:PEF_index+time_after]
		energy_log = energy_log[PEF_index-time_before_PEF:PEF_index+time_after]

		#==================vortex range===============
		min_f_a = 275
		max_f_a = 2200
		#max_f_a = 3000 # for digidoc
		P_vortex = np.log(P[(P_f>min_f_a) &(P_f<max_f_a),PEF_index-time_before_PEF:PEF_index+time_after])
		P_vortex_max, vortex_freq = follow_peak_in_band(P_vortex,energy,1.0,time_before_PEF)

		vortex_freq = vortex_freq*fs/P_NFFT+min_f_a
		vortex_freq_norm = (vortex_freq-np.min(vortex_freq))/(np.max(vortex_freq)-np.min(vortex_freq))
		vortex_freq_norm[vortex_freq_norm==0]=np.nan
		# vortex_freq[vortex_freq==min_f_a]=0

		axa = plt.subplot(3,1,3)
		Ptmp = P_vortex.copy()
		Ptmp[P_vortex_max==True] = -5
		axa.pcolorfast(P_t[PEF_index-time_before_PEF:PEF_index+time_after], P_f[(P_f>min_f_a) &(P_f<max_f_a)], Ptmp, cmap=plt.cm.bone)

		#==================side whistle low===============
		min_f_a = 1500
		max_f_a = 2000
		P_s_low = np.log(P[(P_f>min_f_a) &(P_f<max_f_a),PEF_index-time_before_PEF:PEF_index+time_after])

		first = np.nan
		second = np.nan
		for i in range(len(vortex_freq)):
		    if np.isnan(vortex_freq_norm[i]) and np.isnan(first):
		        first = i
		        print first,
		        
		    if not np.isnan(vortex_freq_norm[i]):
		        second = i
		        
		P_s_low_max1, s_low_freq1 = inward_peak_in_band(P_s_low,energy,1.0,first+100)
		P_s_low_max2, s_low_freq2 = outward_peak_in_band(P_s_low,energy,1.0,second-20)
		P_s_low_max = P_s_low_max1 | P_s_low_max2

		s_low_freq = np.zeros(s_low_freq1.shape)
		for i in range(s_low_freq1.shape[0]):
			s_low_freq[i] = np.argmax(P_s_low_max[:,i])

		s_low_freq = s_low_freq*fs/P_NFFT+min_f_a
		s_low_freq_norm = (s_low_freq-np.min(s_low_freq))/(np.max(s_low_freq)-np.min(s_low_freq))
		s_low_freq_norm[s_low_freq_norm==0]=np.nan
		# s_low_freq[s_low_freq==min_f_a]=0

		ax = plt.subplot(3,1,2,sharex=axa)
		Ptmp = P_s_low.copy()
		Ptmp[P_s_low_max==True] = -5
		ax.pcolorfast(P_t[PEF_index-time_before_PEF:PEF_index+time_after], P_f[(P_f>min_f_a) &(P_f<max_f_a)], Ptmp, cmap=plt.cm.bone)
		

		#==================side whistle high===============
		min_f_a = 5000
		max_f_a = 5500
		P_s_hi = np.log(P[(P_f>min_f_a) &(P_f<max_f_a),PEF_index-time_before_PEF:PEF_index+time_after])
		P_s_hi_max, s_hi_freq = follow_peak_in_band(P_s_hi,energy,1.0,time_before_PEF)

		s_hi_freq = s_hi_freq*fs/P_NFFT+min_f_a
		s_hi_freq_norm = (s_hi_freq-np.min(s_hi_freq))/(np.max(s_hi_freq)-np.min(s_hi_freq))
		s_hi_freq_norm[s_hi_freq_norm==0]=np.nan

		ax = plt.subplot(3,1,1,sharex=axa)
		Ptmp = P_s_hi.copy()
		Ptmp[P_s_hi_max==True] = -5
		ax.pcolorfast(P_t[PEF_index-time_before_PEF:PEF_index+time_after], P_f[(P_f>min_f_a) &(P_f<max_f_a)], Ptmp, cmap=plt.cm.bone)


		
		plt.show()

		# custom add graphic
		f1 = plt.figure(3)
		f1.clear()
		plt.ion()
		plt.subplot(211)
		plt.plot(vortex_freq_norm,label='vortex')
		plt.plot(s_low_freq_norm,label='Side Fundamental')
		plt.plot(s_hi_freq_norm,label='Side Harmonic')

		plt.subplot(212)
		plt.plot(P_t[PEF_index-time_before_PEF:PEF_index+time_after],energy)
		#plt.plot(P_t[PEF_index-time_before_PEF:PEF_index+time_after],energy_log)


		plt.legend()
		plt.show()

		print "Done!"

		x1 = vortex_freq
		x2 = s_low_freq
		x3 = s_hi_freq

		# make same sampling rate
		f = resample(f,len(f)/fs_w*fs_effective)

		x1[np.isnan(x1)] = 0
		x2[np.isnan(x2)] = 0
		x3[np.isnan(x3)] = 0

		x1_m = np.argmax(x1)
		f_m = np.argmax(f)

		# align maxima
		if x1_m>f_m:
		    f = np.hstack((np.zeros((x1_m-f_m,)),f))
		else:
		    x1 = np.hstack((np.zeros((f_m-x1_m,)),x1))
		    x2 = np.hstack((np.zeros((f_m-x1_m,)),x2))
		    x3 = np.hstack((np.zeros((f_m-x1_m,)),x3))
		    
		# make equal lengths
		if len(x1)>len(f):
		    f = np.hstack((f,np.zeros((len(x1)-len(f),))))
		else:
		    x1 = np.hstack((x1,np.zeros((len(f)-len(x1),))))
		    x2 = np.hstack((x2,np.zeros((len(f)-len(x2),))))
		    x3 = np.hstack((x3,np.zeros((len(f)-len(x3),))))
		    
		v = np.cumsum(f/fs_effective)
		df = pd.DataFrame(np.vstack((f,v,x1,x2,x3)).T,columns=['flow','volume','vortex','side_f','side_h1'])

		df['PWG'] =str(waveform['Header']['Group'])+str(waveform['Header']['Name'])
		df['meta'] = meta

		#df.to_csv('Results/'+str(waveform['Header']['Group'])+str(waveform['Header']['Name'])+'.csv')

		#df[['vortex','side_f','side_h1']].plot()
		plt.figure()
		df['flow'].plot()

	
	def cubic_peak_interpolation(self,P_slice):
		X_inv = .5*np.array([[1, -2, 1],[-1, 0, 1]])
		# Get the peak index
		peak_idx = min(max(np.argmax(P_slice),1), len(P_slice)-2)

		# Now perform cubic interpolation to get sub-bin accuracy (we assume the peak is never the first or last index)
		Z = np.dot(X_inv, P_slice[peak_idx-1:peak_idx+2])

		# We now have the a and b coefficients of the parabola (we disregard c), so we find peak offset
		# and use that to get sub-bin accuracy:
		return peak_idx - Z[1]/(2*Z[0])

	# Weigh lower frequencies better
	def decreasing_windowed_fft_analysis(self, P, sigma=10.0, lowweight=.9, fs=2):
		max_idx = np.unravel_index(np.argmax(P), P.shape)

		# Step forward and backward
		fft_f = np.zeros((P.shape[1],))
		fft_f[max_idx[1]] = max_idx[0]
		for t in xrange(max_idx[1]+1, P.shape[1]):
		    # Build an appropriate window centered on fft_f[t-1]
		    window = np.exp(-(fft_f[t-1]*lowweight - np.arange(P.shape[0]))**2/sigma**2)

		    fft_f[t] = self.cubic_peak_interpolation(P[:,t]*window)

		for t in xrange(max_idx[1]-1, 0, -1):
		    # Build an appropriate window centered on fft_f[t+1]
		    window = np.exp(-(fft_f[t+1]*lowweight - np.arange(P.shape[0]))**2/sigma**2)

		    fft_f[t] = self.cubic_peak_interpolation(P[:,t]*window)

		return fft_f*fs/(2*P.shape[0])



