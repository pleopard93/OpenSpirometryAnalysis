from Tkinter import *
import json
from graph1 import graph1
from graph2 import graph2

class openSpiro:
	def __init__(self,  window):
		"""
		Sets up initial window fields
		"""
		self.window = window

		self.layoutWindow()


	def layoutWindow(self):
		# Create and add file label
		self.fileLabel = Label(topFrame, text="Filename").pack(side=LEFT, anchor=NW, fill=X, expand=NO)

		# Create and add file entry
		self.fileEntry = Entry(topFrame)
		self.fileEntry.pack(side=LEFT, anchor=NW, fill=X, expand=NO)
		self.fileEntry.insert(0, "001.json")

		# Create and add button
		self.b1 = Button(topFrame, text='Load', command=lambda: self.beginAnalysis(self.getFileName())).pack(side=LEFT, anchor=NW, fill=X, expand=NO)

		# Create listboxes
		self.testListBox = Listbox(middleFrame1, width=50, height=6)
		self.effortsListBox = Listbox(middleFrame1, width=4, height=6)

		# Pack listboxes
		self.testListBox.pack(side=LEFT, fill=BOTH, anchor=W, expand=YES)
		self.effortsListBox.pack(side=LEFT, fill=BOTH, anchor=E, expand=YES)

		# Create text labels
		self.l1 = Label(middleFrame2, text="Completion time: ")
		self.l2 = Label(middleFrame2, text="FEVOne / FVC: ")
		self.l3 = Label(middleFrame2, text="FVC in Liters: ")
		self.l4 = Label(middleFrame2, text="FEVOne in Liters: ")

		# Add text labels
		self.l1.pack(side=TOP, anchor=N, fill=X, expand=YES)
		self.l2.pack(side=TOP, anchor=N, fill=X, expand=YES)
		self.l3.pack(side=TOP, anchor=N, fill=X, expand=YES)
		self.l4.pack(side=TOP, anchor=N, fill=X, expand=YES)

		self.graph1 = graph1(bottomFrame)
		self.graph2 = graph2(bottomFrame)

		# Pack frames
		topFrame.pack(side=TOP, anchor=NW, expand=YES)
		middleFrame1.pack(side=TOP, fill=BOTH, expand=YES)
		middleFrame2.pack(side=TOP, fill=BOTH, expand=YES)
		bottomFrame.pack(side=BOTTOM, fill=BOTH, anchor=SW, expand=YES)


	def getFileName(self):
		return str(self.fileEntry.get())


	def showTestVariables(self, data):
		"""
		Shows test values in window
		"""
		self.l1.config(text="Completion time: " + data["Completion"])
		self.l2.config(text="FEVOne / FVC: " + repr(data["Efforts"][0]["FEVOneOverFVC"]))
		self.l3.config(text="FVC in Liters: " + repr(data["Efforts"][0]["FVCInLiters"]))
		self.l4.config(text="FEVOne in Liters: " + repr(data["Efforts"][0]["FVCInLiters"]))


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
			# Get selected line index
			index = self.testListBox.curselection()[0]
			self.showEfforts(data["Tests"][index])

		# Clear listbox
		self.testListBox.delete(0, END)

		for index, test in enumerate(data["Tests"]):
			# Add test mouthpiece and downstream tube to list box
			# Add 1 to the index to soothe the OCD
			self.testListBox.insert(END, repr(index+1) + ".) " + test["Mouthpiece"] + " - " + test["DownstreamTube"])

		# Left mouse click on a list item to display selection
		self.testListBox.bind('<ButtonRelease-1>', handleTestSelection)


	def showEfforts(self, data):
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
			self.showTestVariables(data)
			self.graph1.showGraph(data["Efforts"][index])
			self.graph2.showGraph(data["Efforts"][index])

		# Clear listbox
		self.effortsListBox.delete(0, END)

		for index, effort in enumerate(data["Efforts"]):
			# Add test mouthpiece and downstream tube to list box
			# Add 1 to the index to soothe the OCD
			self.effortsListBox.insert(END, index+1)

		# Left mouse click on a list item to display selection
		self.effortsListBox.bind('<ButtonRelease-1>', handleEffortSelection)

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
