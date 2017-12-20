import tkinter
from tkinter import filedialog
from PIL import ImageTk, Image
from tkinter import ttk
import tkinter.messagebox

import os
import sys


import BB_To_Tree
import matplotlib.pyplot as plt


import SSD_Core as core

class MainWindow:
	def __init__(self):
		print ('initializing...')
		
		self.mainHandle = tkinter.Tk()
		#self.mainHandle.configure(background='black')
		self.mainHandle.title('Mathematical Expression Recognition - Demo')
		self.mainHandle.geometry("850x700")

		self.camvas_border = 10
		
		self.initGUI()

		try:
			os.mkdir('temp')
		except:
			pass
			
		self.BBparser = BB_To_Tree.BBParser()
		
		self.ssd = core.SSD_Core()
		
	def initGUI(self):
	
		############# BROWSE #############################
	
		file_frame = tkinter.Frame(master=self.mainHandle, bd = 10)
		file_frame.pack(fill=tkinter.BOTH)
		
		self.f1Entry = tkinter.Entry(file_frame, bd = 5, width= 50)
		self.f1Entry.pack(side = tkinter.LEFT)
		
		f1Browse = tkinter.Button(file_frame, text = "Browse Folder", width=15, command=self.open_callback)
		f1Browse.pack(side = tkinter.LEFT)

		############# IMG FRAME #############################

		img_frame = tkinter.Frame(master=self.mainHandle, bd = 0)
		img_frame.pack(fill=tkinter.X)
	
		

		self.rawimg = ImageTk.PhotoImage(Image.new('RGB', (300, 300)))
		self.rawimg_canvas = tkinter.Canvas(img_frame, height=300, width=300, borderwidth=self.camvas_border, relief="raised")
		self.rawimg_canvas.pack(side = tkinter.LEFT)
		self.image_on_canvas_raw = self.rawimg_canvas.create_image(self.camvas_border, self.camvas_border, anchor = tkinter.NW, image = self.rawimg)
		
		
		self.processedimg = ImageTk.PhotoImage(Image.new('RGB', (300, 300)))
		self.processed_canvas = tkinter.Canvas(img_frame, height=300, width=300, borderwidth=self.camvas_border, relief="raised")
		self.processed_canvas.pack(side = tkinter.LEFT)
		self.image_on_canvas_processed = self.processed_canvas.create_image(self.camvas_border, self.camvas_border, anchor = tkinter.NW, image = self.processedimg)
		
		
		#----------------------------

		scrollbox_frame = tkinter.Frame(img_frame, bd = 0)
		scrollbox_frame.pack(fill=tkinter.Y)

		scrollbar = tkinter.Scrollbar(scrollbox_frame)
		scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)

		self.filelistbox = tkinter.Listbox(scrollbox_frame, height=20, width = 30)
		self.filelistbox.pack()

		self.filelistbox.config(yscrollcommand=scrollbar.set)
		scrollbar.config(command=self.filelistbox.yview)
		self.filelistbox.bind('<<ListboxSelect>>', self.onselectfile)

		############# LATEX FRAME #############################
		
		latex_frame = tkinter.Frame(master=self.mainHandle, bd = 0)
		latex_frame.pack(fill=tkinter.X)

		self.lateximg = ImageTk.PhotoImage(Image.new('RGB', (600, 300)))
		self.lateximg_canvas = tkinter.Canvas(latex_frame, height=300, width=600, borderwidth=self.camvas_border, relief="raised")
		self.lateximg_canvas.pack(side = tkinter.LEFT)
		self.image_on_canvas_latex = self.lateximg_canvas.create_image(self.camvas_border, self.camvas_border, anchor = tkinter.NW, image = self.lateximg)
	
	def Test(self):
		
		with open('ssd_train.txt') as f:
			z = f.readlines()
			self.test_candidate = z
		
		
	def onselectfile(self, evt):
		w = evt.widget
		index = int(w.curselection()[0])
		fname = w.get(index)
		
		#self.rawimg_canvas.delete(self.image_on_canvas_raw)
		self.rawimg = ImageTk.PhotoImage(Image.open(self.file_path + fname))
		self.rawimg_canvas.itemconfig(self.image_on_canvas_raw, image = self.rawimg)
		
		
		########### FOR DEBUG #####################
		
		if False:
			for i in self.test_candidate:
				if fname in i:
					BB_list_Str = i
		
		###########################################
		
		BB_list_Str = self.ssd.generatePrediction(self.file_path + fname)
		self.processedimg = ImageTk.PhotoImage(Image.open('./temp/predict_temp.png'))
		self.processed_canvas.itemconfig(self.image_on_canvas_processed, image = self.processedimg)

		
		latex_string = self.handleBBList(BB_list_Str)
		
		latex_string = latex_string.replace('\\\\', r'\\')
		
		try:
			self.showLatex(latex_string)
		except ValueError as e:
			print('Wrong parse!')
			plt.clf()
		
		print(latex_string)
		
		
	def handleBBList(self, raw_BB):
		return self.BBparser.process(raw_BB)

	def showLatex(self, latex_string):
		plt.text(0.5, 0.5, r'$' + latex_string + r'$',fontsize=30, horizontalalignment = 'center')

		#hide axes
		fig = plt.gca()
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)

		figure = plt.gcf() # get current figure
		figure.set_size_inches(6, 3)


		fig.spines['top'].set_visible(False)
		fig.spines['right'].set_visible(False)
		fig.spines['bottom'].set_visible(False)
		fig.spines['left'].set_visible(False)

		plt.savefig('./temp/temp.png')
		plt.clf()
		
		
		self.lateximg_canvas.delete(self.image_on_canvas_latex)
		self.lateximg = ImageTk.PhotoImage(Image.open('./temp/temp.png'))
		
		self.image_on_canvas_latex = self.lateximg_canvas.create_image(self.camvas_border, self.camvas_border, anchor = tkinter.NW, image = self.lateximg)
		
		#self.lateximg_canvas.itemconfig(self.image_on_canvas_latex, image = self.lateximg)
		
	def open_callback(self):
		self.file_path = filedialog.askdirectory(initialdir='.') + '/'
		#self.box_remaining.delete(0,tkinter.END)
		#self.box_remaining.insert(0,file_path)
		
		for root, dirs, files in os.walk(self.file_path):
			#self.pending_file_list = files
			#self.box_remaining['values'] = self.pending_file_list
			self.filelistbox.delete(0, tkinter.END)
			for i in files:
				self.filelistbox.insert(tkinter.END, i)
			self.current_folder = root + '/'
			break

	def run(self):
		self.mainHandle.mainloop()
		
	def close(self, event=None):
		self.mainHandle.destroy()

instance = MainWindow()

#instance.Test()

instance.run()