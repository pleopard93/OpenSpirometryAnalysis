from Tkinter import *
import json
from os import listdir
from os.path import isfile, join
from graph1 import graph1
from graph2 import graph2
import pdb

labelText = ["Completion Time:","Effort Notes:","Test Notes:","Group:","PWG Params:"]

class openSpiro:
	def __init__(self,  window):
		"""
		Sets up initial window fields
		"""
		self.window = window

		self.layoutWindow()
		self.showFileNames()


	def layoutWindow(self):
		# Create and add file label
		self.fileLabel = Label(topFrame, text="File Path").pack(side=LEFT, anchor=NW, fill=X, expand=NO)

		# Create and add file entry
		self.fileEntry = Entry(topFrame)
		self.fileEntry.pack(side=LEFT, anchor=NW, fill=X, expand=NO)
		self.fileEntry.insert(0, "../audio_curve_data/")
		self.audioJSONDirectory = "../audio_curve_data/"

		# Create and add button
		self.b1 = Button(topFrame, text='Load', command=self.buttonLoadPressed).pack(side=LEFT, anchor=NW, fill=X, expand=NO)

		# Create listboxes
		self.filesListBox = Listbox(middleFrame1, width=50, height=6)
		self.testListBox = Listbox(middleFrame1, width=50, height=6)
		self.effortsListBox = Listbox(middleFrame1, width=4, height=6)

		# Pack listboxes
		self.filesListBox.pack(side=LEFT, fill=BOTH, anchor=W, expand=YES)
		self.testListBox.pack(side=LEFT, fill=BOTH, anchor=W, expand=YES)
		self.effortsListBox.pack(side=LEFT, fill=BOTH, anchor=E, expand=YES)

		# Create text labels
		self.l1 = Label(middleFrame2, text=labelText[0])
		self.l2 = Label(middleFrame2, text=labelText[1])
		self.l3 = Label(middleFrame2, text=labelText[2])
		self.l4 = Label(middleFrame2, text=labelText[3])
		self.l5 = Label(middleFrame2, text=labelText[4])

		# Add text labels
		self.l1.pack(side=TOP, anchor=N, fill=X, expand=YES)
		self.l2.pack(side=TOP, anchor=N, fill=X, expand=YES)
		self.l3.pack(side=TOP, anchor=N, fill=X, expand=YES)
		self.l4.pack(side=TOP, anchor=N, fill=X, expand=YES)
		self.l5.pack(side=TOP, anchor=N, fill=X, expand=YES)

		self.graph1 = graph1(bottomFrame)
		self.graph2 = graph2(bottomFrame)

		# Pack frames
		topFrame.pack(side=TOP, anchor=NW, expand=YES)
		middleFrame1.pack(side=TOP, fill=BOTH, expand=YES)
		middleFrame2.pack(side=TOP, fill=BOTH, expand=YES)
		bottomFrame.pack(side=BOTTOM, fill=BOTH, anchor=SW, expand=YES)

		self.waveform = {}


	def getFileName(self):
		return str(self.fileEntry.get())

	def buttonLoadPressed(self):
		self.audioJSONDirectory = self.getFileName()
		self.showFileNames(self.audioJSONDirectory)

	def clearFields(self, willClearTests, willClearEfforts):
		"""
		Clear necessary fields depending on updated selection
		"""
		self.graph1.clearGraph()
		self.graph2.clearGraph()

		# Clear listboxes
		if willClearTests:
			self.testListBox.delete(0, END)

		if willClearEfforts:
			self.effortsListBox.delete(0, END)

		# Clear text labels
		self.l1.config(text=labelText[0])
		self.l2.config(text=labelText[1])
		self.l3.config(text=labelText[2])
		self.l4.config(text=labelText[3])
		self.l5.config(text=labelText[4])


	def showFileNames(self, filePath="../"):
		# Clear all fields before showing new files
		self.clearFields(True, True)

		files = [f for f in listdir(filePath) if isfile(join(filePath, f)) and ".json" in f]

		def handleTestSelection(event):
			"""
			Reads the listbox selection
			"""
			self.clearFields(False, True)
			# Get selected line index
			index = self.filesListBox.curselection()[0]

			if ".json" in files[index]:
				self.beginAnalysis(filePath + "/" + files[index])

		# Clear listbox
		self.filesListBox.delete(0, END)

		for file in enumerate(files):
			# Add file name to listbox if .json
			if ".json" in file[1]:
				self.filesListBox.insert(END, file[1])

		# Left mouse click on a list item to display selection
		self.filesListBox.bind('<ButtonRelease-1>', handleTestSelection)

	def showTestVariables(self, data, metadata):
		"""
		Shows test values in window
		"""
		index = self.effortsListBox.curselection()[0]

		effortNotesData = 'No Effort Notes'
		if "Notes" in data["Efforts"][index].keys():
			effortNotesData = repr(data["Efforts"][index]["Notes"])

		testNotesData = 'No Test Notes'
		if "Notes" in data.keys():
			testNotesData = repr(data["Notes"])

		if "CompletionFormatted" in data:
			self.l1.config(text=labelText[0] + ' ' + data["CompletionFormatted"])
		else:
			self.l1.config(text=labelText[0] + ' unknown')

		
		self.l2.config(text=labelText[1] + ' ' + effortNotesData)
		self.l3.config(text=labelText[2] + ' ' + testNotesData)
		self.l4.config(text=labelText[3] + ' ' + metadata["UserGroup"])

		if self.waveform is not None:
			self.l5.config(text=labelText[4] + ' ' + str(self.waveform["Parameters"]))
		else:
			self.l5.config(text=labelText[4] + ' None')



	def beginAnalysis(self, filename):
		"""
		Called from text entry callbacks. Calls data display functions
		"""
		data = self.loadTestData(filename)
		self.showTests(data)


	def loadTestData(self, filename):
		"""
		Loads test data from JSON file and returns
		"""

		with open(filename) as data_file:
			data = json.load(data_file)

		return data


	def showTests(self, data):
		"""
		Show the combinations of mouthpiece and downstream tube for each test.
		When user selects a combination, the data will populate and show efforts.
		"""
		def handleTestSelection(event):
			"""
			Reads the listbox selection
			"""
			self.clearFields(False, False)

			# Get selected line index
			index = self.testListBox.curselection()[0]
			self.showEfforts(data["Tests"][index],data["Metadata"])

		# Clear listbox
		self.testListBox.delete(0, END)

		for index, test in enumerate(data["Tests"]):
			# Add test mouthpiece and downstream tube to list box
			# Add 1 to the index to soothe the OCD
			if "DownstreamTube" in test.keys():
				self.testListBox.insert(END, repr(index+1) + ".) " + test["Mouthpiece"] + " - " + test["DownstreamTube"])
			else:
				self.testListBox.insert(END, repr(index+1) + ".) " + test["Mouthpiece"] )

		# Left mouse click on a list item to display selection
		self.testListBox.bind('<ButtonRelease-1>', handleTestSelection)


	def showEfforts(self, data, metadata):
		"""
		Show the list of efforts for each mouthpiece-downstream tube combination.
		When user selects an effort, the test results will show.
		"""
		def handleEffortSelection(event):
			"""
			Reads the listbox selection
			"""
			# Get selected line index
			index = self.effortsListBox.curselection()[0]
			self.waveform = None
			if "PWGFile" in data.keys():
				self.waveform = self.readWaveformData(data["PWGFile"])
			self.showTestVariables(data, metadata)
			self.graph1.showGraph(data["Efforts"][index],self.audioJSONDirectory)
			self.graph2.showGraph(data["Efforts"][index],self.audioJSONDirectory, self.waveform)

		# Clear listbox
		self.effortsListBox.delete(0, END)

		for index, effort in enumerate(data["Efforts"]):
			# Add test mouthpiece and downstream tube to list box
			# Add 1 to the index to soothe the OCD
			self.effortsListBox.insert(END, index+1)

		# Left mouse click on a list item to display selection
		self.effortsListBox.bind('<ButtonRelease-1>', handleEffortSelection)

	def readWaveformData(self, atsfile):

		if atsfile is None: # not a PWG waveform
			return None

		filepath = self.convertFileToFilePath(atsfile)

		waveform_data = {}
		waveform_data["Header"]={}
		waveform_data["Parameters"]={}
		waveform_data["Data"]=[]
		currSection = ""
		with open(filepath) as f:
		    data = f.readlines()
		    for dline in data:
		        sectionChanged = False
		        # know the current section
		        for sec in ["Header","Parameters","Data"]:
		            if dline.find(sec) != -1:
		                currSection = sec
		                sectionChanged = True 
		        if sectionChanged:
		            continue 
		            
		        dline = dline.strip()
		        if (currSection=="Header" or currSection=="Parameters") and len(dline)>1:
		            keyval = dline.split("=")
		            if keyval[1][0].isdigit():
		                keyval[1] = float(keyval[1])
		            waveform_data[currSection][keyval[0]] = keyval[1]
		        elif currSection=="Data" and len(dline)>1:
		            waveform_data[currSection].append(float(dline))
		return waveform_data


	def convertFileToFilePath(self, atsfile):
		prefix = "Waveform/"

		if atsfile.find("ATS24.") >= 0:
			prefix += "ATS24/"
			tmp = atsfile.split(".")
			prefix += "24%02d.wf"%(int(tmp[1]))

		elif atsfile.find("ATS24*.") >= 0:
			prefix += "ATS24v2/"
			tmp = atsfile.split(".")
			prefix += "24%02d.wf"%(int(tmp[1]))

		elif atsfile.find("ATS26.") >= 0:
			prefix += "ATS26/"
			tmp = atsfile.split(".")
			prefix += "26%02d.wf"%(int(tmp[1]))

		elif atsfile.find("Prof") >= 0:
			prefix += "ISO23747/"
			prefix += atsfile + ".wf"

		elif atsfile.find("ISO2678") >= 0:
			prefix += "ISO26782/"
			tmp = atsfile.split(".")
			prefix += "26782%02d.wf"%(int(tmp[1]))

		elif atsfile.find("Custom") >= 0:
			tmp = atsfile.split(".")
			prefix += "Custom%02d.wf"%(int(tmp[1]))
		else:
			return None

		return prefix                       

    

# Create window and set title
window = Tk()
window.wm_title("Open Spiro")

# Create individual frames
topFrame = Frame(window)
middleFrame1 = Frame(window)
middleFrame2 = Frame(window)
bottomFrame = Frame(window)

# Begin app
openSpiro(window)
window.mainloop()
