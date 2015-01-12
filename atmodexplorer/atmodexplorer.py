#Cut and paste from Matplotlib example for embedding GUI in PyQT 
import sys
import os
import random
from matplotlib.backends import backend_qt4
import matplotlib.widgets as widgets
import matplotlib.axes
from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import SIGNAL,SLOT

#Main imports
import numpy as np
import numpy
#import pandas as pd
import sys, pdb
import datetime
#sys.path.append('/home/liamk/mirror/Projects/geospacepy')
#import special_datetime, lmk_utils, satplottools #Datetime conversion class

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib import ticker
from matplotlib.colors import LogNorm, Normalize
import msispy
from collections import OrderedDict

class canvasClick(object):
	def __init__(self,ch,event):
		"""An instance is spawned by canvasClickHandler every time the user clicks on the canvas. 
			Tracks click locations and which axes they occur in"""
		self.ch = ch #Click Handler
		#Pull information from the event which caused this Click object to be created
		self.ax = event.inaxes #Which axes the click occured in
		if self.ax is None:
			raise RuntimeError('A canvasClick object should only be spawned when the triggering click occurs in a valid Matplotlib axes')

		#Click location
		self.x = event.xdata 
		self.y = event.ydata 
		
		#Associated data 
		self.xdata = self.ch.x
		self.ydata = self.ch.y
		self.zdata = self.ch.z

		#Figure out which point was closest to this click
		
		#Compute the Euclidian distance between all points in the userdata and the clicked one
		dist = np.sqrt((self.xdata-self.x)**2+(self.ydata-self.y)**2)

		self.ind = dist.argmin() #index of the closest point to the click location 
		
		self.clx = self.xdata[self.ind]#Closest datapoint x coordinate in clickhandler.user_xdata
		self.cly = self.ydata[self.ind]#Closest datapoint y coordinate in clickhandler.user_ydata
		self.clz = self.zdata[self.ind]#Closest datapoint z coordinate in clickhandler.user_ydata

		#Plot point at clicked location and add resulting lines.line2D to the 'clicked' list
		#Add the plotted line to the clicked list
		self.lines = self.ax.plot(self.clx,self.cly,'mo',alpha=.8)

		#Show a message for each click in the app status bar
		self.ch.canvas.aw.statusBar().showMessage("%s:%f, %s:%f, %s:%f" % (self.ch.canvas.controlstate['xvar'],self.clx,
																			self.ch.canvas.controlstate['yvar'],self.cly,
																			self.ch.canvas.controlstate['zvar'],self.clz), 2000)


	def __str__(self):
		return "Click pxl: (%d,%d), Click closest: (%.2f,%.2f,%.2f), click axes: %s, click index: %d" % (self.x,self.y,\
			self.clx,self.cly,self.clz,str(self.ax),self.ind) 

	def asTuple(self):
		return self.ind,self.x,self.y,self.clx,self.cly,self.clz

	def set_visible(self,val):
		"""Turn the dot highlighting this click location invisible"""
		for ln in self.lines:
			ln.set_visible(val)

class canvasClickHandler(object):
	def __init__(self, canvas):
		"""Takes a canvas object, connects itself to mouseclick events
and figures out where the click was located, and the closest point to the click location
in the canvas object's plot data"""
		self.canvas = canvas
		self.x = canvas.pdh.x.flatten()
		self.y = canvas.pdh.y.flatten()
		self.z = canvas.pdh.z.flatten()
		self.cid = canvas.mpl_connect('button_press_event', self)
		self.clicks = [] #List of canvasClick objects
		self.saved_clicks = dict() #Save clicks by dict key
			
		
	def __call__(self, event):
		print 'click', event
		
		#Handle not being in the axes
		if event.inaxes is None: 
			return

		if len(self.clicks) > 0:
			#Turn last clicked location invisible
			self.clicks[-1].set_visible(False)

		#Create a canvasClick object to track this click
		thisclick = canvasClick(self,event)
		
		#Add to the sequential list of clicks
		self.clicks.append(thisclick)

		#Redraw the canvas
		self.canvas.draw()

	def reset(self):
		#Basically just calls __init__ again, but without rebinding the events
		#Refreshes the connection to the userdata and clears out old clicks
		#so we don't get any wierdness when the data changes
		if self.canvas.pdh.x is not None:
			self.x = self.canvas.pdh.x.flatten()
		if self.canvas.pdh.y is not None:
			self.y = self.canvas.pdh.y.flatten()
		if self.canvas.pdh.z is not None:
			self.z = self.canvas.pdh.z.flatten()
		
		self.clicks = [] #List of canvasClick objects
		self.saved_clicks = dict() #A dictionary of click objects saved by calling canvasClickHandler.save 

	def save(self,key):
		"""Saves last clicked value and closest point into self.saved dictionary"""
		self.saved_clicks[key] = self.clicks[-1]
		print(self.saved_clicks[key])

	def lastclick(self,ax=None):
		"""Returns index,click x, click y, closest data x, closest data y for last click registered by clickHandler"""
		return self.clicks[-1]

	def load(self,key):
		"""Retrieves click location / closest value saved into self.saved"""
		return self.saved_clicks[key]

class MsisRun(object):
	""" Class for individual calls to NRLMSISE00 """
	import msispy
	from collections import OrderedDict

	def __init__(self):
		"""Start with a blank slate"""
		self.dt = datetime.datetime(2000,6,21,12,0,0)
		#Now we make links to these properties/attributes in 
		#the cannocical 'vars' dictionary, which has 
		#keys which are used to populate the combobox widgets,
		self.vars = OrderedDict()
		self.vars['Latitude']=None
		self.vars['Longitude']=None
		self.vars['Altitude']=None
		self.lims = OrderedDict()
		self.lims['Latitude']=[-90.,90.]
		self.lims['Longitude']=[-180.,180.]
		self.lims['Altitude']=[0.,1000.]
		self.shape = None #Tells how to produce gridded output, or use column vectors if no grids
		self.npts = None #Tells how many total points
		

	def __call__(self):
		"""When the run is called, it populates itself with data"""

		print "Now running NRLMSISE00 for %s...\n" % (self.dt.strftime('%c'))
		#Make sure that everything has the same shape
		if numpy.isscalar(self.vars['Latitude']):
			self.vars['Latitude'] = numpy.ones(self.shape)*self.vars['Latitude']
		if numpy.isscalar(self.vars['Longitude']):
			self.vars['Longitude'] = numpy.ones(self.shape)*self.vars['Longitude']
		if numpy.isscalar(self.vars['Altitude']):
			self.vars['Altitude'] = numpy.ones(self.shape)*self.vars['Altitude']

		#Now flatten everything to make the msis call
		lat=self.vars['Latitude'].flatten()
		lon=self.vars['Longitude'].flatten()
		alt=self.vars['Altitude'].flatten()
		self.npts = len(lat)
		self.species,self.t_exo,self.t_alt,self.drivers = msispy.msis(self.dt,lat,lon,alt)
		
		#Now add temperature the variables dictionary
		self.vars['T_exospheric'] = self.t_exo
		self.vars['Temperature'] = self.t_alt
		
		#Now add all of the different number and mass densities as
		#references to to the vars dictionary
		for s in self.species:
			self.vars[s] = self.species[s]

		#Now make everything into the appropriate shape, if were
		#expecting grids. Otherwise make everything into a column vector
		if self.shape is None:
			self.shape = (self.npts,1)
			
		for v in self.vars:
			self.vars[v] = numpy.reshape(self.vars[v],self.shape)
			if v not in ['Latitude','Longitude','Altitude']:
				self.lims[v] = [numpy.nanmin(self.vars[v].flatten()),numpy.nanmax(self.vars[v].flatten())]
			#print "lims for var %s = [%f,%f]" % (v,self.lims[v][0],self.lims[v][1])		

class ModelRunner(object):
	""" Makes model calls """
	def __init__(self,canvas):
		
		self.cv = canvas

		#Init the runs list (holds each run in sequence)
		self.runs = []
		#Start with a blank msis run
		self.nextrun = MsisRun()
		self.nextrun.dt = datetime.datetime(2000,6,21,12,0,0) #Summer solstice
		self.n_total_runs=0
		self.n_max_runs=10
		#Create the dictionary that stores the settings for the next run
		
	def __call__(self):
		#Add to runs list, create a new nextrun
		self.runs.append(self.nextrun)
		self.nextrun = MsisRun()
		self.n_total_runs+=1
		print "Model run number %d added." % (self.n_total_runs)
		if len(self.runs)>self.n_max_runs:
			del self.runs[0]
			print "Exceeded total number of stored runs %d. Run %d removed" %(self.n_max_runs,self.n_total_runs-self.n_max_runs)


class plotDataHandler(object):
	def __init__(self,canvas,plottype='line',cscale='linear',mapproj='moll'):
		"""
		Takes a singleMplCanvas instance to associate with
		and plot on
		"""
		self.canvas = canvas
		self.fig = canvas.fig
		self.ax = canvas.ax
		self.axpos = canvas.ax.get_position()
		pts = self.axpos.get_points().flatten() # Get the extent of the bounding box for the canvas axes
		self.cbpos = [.1,pts[1]-.15,.8,.1]
		self.cb = None
		self.map = None #Place holder for map instance if we're plotting a map
		#The idea is to have all plot settings described in this class,
		#so that if we want to add a new plot type, we only need to modify 
		#this class
		self.plottypes = dict()
		self.plottypes['line'] = {'gridxy':False}
		self.plottypes['pcolor'] = {'gridxy':True}
		self.plottypes['map'] = {'gridxy':True}
		if plottype not in self.plottypes:
			raise ValueError('Invalid plottype %s! Choose from %s' % (plottype,str(self.plottypes.keys())))
		
		#Assign settings from input
		self.plottype = plottype
		self.mapproj = mapproj # Map projection for Basemap

		#Init the data variables/
		self.clear_data()

	def clear_data(self):
		self.x,self.y,self.z = None,None,None #must be numpy.array
		self.xname,self.yname,self.zname = None,None,None #must be string
		self.xbounds,self.ybounds,self.zbounds = None,None,None #must be 2 element tuple
		self.xlog, self.ylog, self.zlog = False, False, False
		self.npts = None

	def associate_data(self,varxyz,vardata,varname,varbounds,varlog):
		#Sanity check 
		thislen = len(vardata.flatten()) 
		thisshape = vardata.shape

		#Check total number of points
		if self.npts is not None:
			if thislen != self.npts:
				raise RuntimeError('Variable %s passed for axes %s had wrong flat length, got %d, expected %d' % (varname,varaxes,
					thislen,self.npts))

		#Check shape
		for v in [self.x,self.y,self.z]:
			if v is not None:
				if v.shape != thisshape:
					raise RuntimeError('Variable %s passed for axes %s had mismatched shape, got %s, expected %s' % (varname,varaxes,
					str(thisshape),str(v.shape)))

		#Parse input and assign approriate variable
		if varxyz in ['x','X',0,'xvar']:
			self.x = vardata
			self.xname = varname
			self.xbounds = varbounds
			self.xlog = varlog
		elif varxyz in ['y','Y',1,'yvar']:
			self.y = vardata
			self.yname = varname
			self.ybounds = varbounds
			self.ylog = varlog
		elif varxyz in ['z','Z',2,'C','c','zvar']:
			self.z = vardata
			self.zname = varname
			self.zbounds = varbounds
			self.zlog = varlog
		else:
			raise ValueError('%s is not a valid axes for plotting!' % (str(varaxes)))

	
	def plot(self,*args,**kwargs):
		self.ax.cla()
		self.ax.set_adjustable('datalim')
		self.map = None #Make sure that we don't leave any maps lying around if we're not plotting maps
		
		#self.zbounds = (numpy.nanmin(self.z),numpy.nanmax(self.z))
		
		if self.zlog:
			self.z[self.z<=0.] = numpy.nan
			norm = LogNorm(vmin=self.zbounds[0],vmax=self.zbounds[1]) 
			locator = ticker.LogLocator()
		else:
			norm = Normalize(vmin=self.zbounds[0],vmax=self.zbounds[1])
	
		if self.plottype == 'line':
			#Plot a simple 2d line plot
			if self.cb is not None:
				self.cb.remove()
				self.cb = None
				#self.ax.set_position(self.axpos)

			self.ax.plot(self.x,self.y,*args,**kwargs)
			self.ax.set_xlabel(self.xname)
			self.ax.set_xlim(self.xbounds)
			if self.xlog:
				self.ax.set_xscale('log',nonposx='clip')
				#self.ax.set_xlim(0,np.log(self.xbounds[1]))
			self.ax.set_ylabel(self.yname)
			self.ax.set_ylim(self.ybounds)
			if self.ylog:
				self.ax.set_yscale('log',nonposx='clip')
				#self.ax.set_ylim(0,np.log(self.ybounds[1]))
			self.ax.set_title('%s vs. %s' % (self.xname,self.yname))
			if not self.xlog and not self.ylog:
				self.ax.set_aspect(1./self.ax.get_data_ratio())
			#try:
			#	self.fig.tight_layout()
			#except:
			#	print "Tight layout for line failed"
		
		elif self.plottype == 'pcolor':

			mappable = self.ax.pcolormesh(self.x,self.y,self.z,norm=norm,shading='gouraud',**kwargs)
			#m.draw()

			self.ax.set_xlabel(self.xname)
			self.ax.set_xlim(self.xbounds)
			if self.xlog:
				self.ax.set_xscale('log',nonposx='clip')
			
			self.ax.set_ylabel(self.yname)
			self.ax.set_ylim(self.ybounds)
			if self.ylog:
				self.ax.set_xscale('log',nonposx='clip')
			
			self.ax.xaxis.tick_top()

			if self.cb is None:
				self.cb = self.fig.colorbar(mappable,ax=self.ax,orientation='horizontal')
			else:
				self.cb.remove()
				self.cb = self.fig.colorbar(mappable,ax=self.ax,orientation='horizontal')
			self.ax.set_position(self.axpos)
			self.cb.ax.set_position(self.cbpos)

			self.cb.set_label(self.zname)
			self.ax.set_aspect(1./self.ax.get_data_ratio())
			self.fig.suptitle('%s vs. %s (color:%s)' % (self.xname,self.yname,self.zname))
			#try:
			#	self.fig.tight_layout() #Makes sure all labels are visible
			#except:
			#	print "Tight layout for pcolor failed"

		elif self.plottype == 'map':
			if self.mapproj=='moll':
				m = Basemap(projection=self.mapproj,llcrnrlat=self.ybounds[0],urcrnrlat=self.ybounds[1],\
					llcrnrlon=self.xbounds[0],urcrnrlon=self.xbounds[1],lat_ts=20,resolution='c',ax=self.ax,lon_0=0.)
			m.drawcoastlines()
			#m.fillcontinents(color='coral',lake_color='aqua')
			# draw parallels and meridians.
			m.drawparallels(numpy.arange(-90.,91.,15.))
			m.drawmeridians(numpy.arange(-180.,181.,30.))
			if self.zlog:
				mappable = m.contour(self.x,self.y,self.z,15,linewidths=1.5,latlon=True,norm=norm,
					vmin=self.zbounds[0],vmax=self.zbounds[1],locator=locator,**kwargs)
			else:
				mappable = m.contour(self.x,self.y,self.z,15,linewidths=1.5,latlon=True,norm=norm,
					vmin=self.zbounds[0],vmax=self.zbounds[1],**kwargs)
				
			#mappable = m.pcolormesh(self.x,self.y,self.z,norm=norm,**kwargs)
			self.ax.xaxis.tick_top()
			latbounds = [self.ybounds[0],self.ybounds[1]]
			lonbounds = [self.xbounds[0],self.xbounds[1]]

			lonbounds[0],latbounds[0] = m(lonbounds[0],latbounds[0])
			lonbounds[1],latbounds[1] = m(lonbounds[1],latbounds[1])

			self.ax.set_ylim(latbounds)
			self.ax.set_xlim(lonbounds)

			#m.draw()
			if self.cb is None:
				self.ax.set_position(self.axpos)
				self.cb = self.fig.colorbar(mappable,ax=self.ax,orientation='horizontal')
			else:
				self.cb.remove()
				self.ax.set_position(self.axpos)
				self.cb = self.fig.colorbar(mappable,ax=self.ax,orientation='horizontal')
			#self.cb.ax.set_position(self.cbpos)
			self.cb.set_label(self.zname)
			self.fig.suptitle('%s projection map of %s' % (self.mapproj,self.zname))
			self.map = m


class BoundsDialog(QtGui.QDialog):
	"""A 2 field dialog box for changing axes boundaries"""
	def __init__(self, parent = None, initialbounds=None):
		super(BoundsDialog, self).__init__(parent)

		layout = QtGui.QVBoxLayout(self)

		# nice widget for editing the date
		self.mainlabel = QtGui.QLabel('Enter Axes Min/Max')
		layout.addWidget(self.mainlabel)

		hl = QtGui.QHBoxLayout()
		self.minline = QtGui.QLineEdit()
		self.maxline = QtGui.QLineEdit()
		hl.addWidget(self.minline)
		hl.addWidget(self.maxline)
		if initialbounds is not None:
			self.minline.setText("%.3f" % (initialbounds[0]))
			self.maxline.setText("%.3f" % (initialbounds[1]))
		layout.addLayout(hl)

		# OK and Cancel buttons
		self.buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,parent=self)
		
		self.buttons.accepted.connect(self.accept)
		self.buttons.rejected.connect(self.reject)
		
		layout.addWidget(self.buttons)

	# get current date and time from the dialog
	def getfloats(self):
		minstr = str(self.minline.text())
		maxstr = str(self.maxline.text())
		return float(minstr),float(maxstr)

	# static method to create the dialog and return (date, time, accepted)
	@staticmethod
	def getMinMax(parent = None,initialbounds=None):
		dialog = BoundsDialog(parent,initialbounds=initialbounds)
		result = dialog.exec_()
		minflt,maxflt = dialog.getfloats()
		return (minflt,maxflt,result == QtGui.QDialog.Accepted)

class singleMplCanvas(FigureCanvas):
	"""This is also ultimately a QWidget and a FigureCanvasAgg"""
	#Prepare the signals that this can emit
	canvasrefreshed = QtCore.pyqtSignal()

	def __init__(self,parent=None,appwindow=None,figsize=(5,4),dpi=200):
		#QWidget stuff
		self.fig = Figure(figsize=figsize, dpi=dpi)
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)

		FigureCanvas.setSizePolicy(self,
								   QtGui.QSizePolicy.Expanding,
								   QtGui.QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)

		#Context menu
		self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

		self.set_mplparams()
		self.ax = self.fig.add_axes([0.1,.3,.8,.6])
		#Create a data handler and a click handler
		self.pdh = plotDataHandler(self)
		
		#Associate with a model runner and the main window (allows access of children of the canvas to the QWidgets)
		self.mr = ModelRunner(self)
		self.aw = appwindow

		#Placeholder for a click handler
		#self.ch = None

		#Create the dictionary of control settings, which are set when
		#a QWidget which is bound to one of the event slots is triggered
		#The initial settings here will be used for the starting state of the GUI
		self.controlstate = {'model_index':-1,'datetime':datetime.datetime(2000,6,21,12,0,0),\
						 'lat':45.,'lon':0.,'alt':85.,\
						 'plottype':'map',\
						 'xvar':'Longitude','xbounds':[-180.,180.],'xnpts':50.,'xlog':False,\
						 'yvar':'Latitude','ybounds':[-90.,90.],'ynpts':50.,'ylog':False,\
						 'zvar':'Temperature','zbounds':None,'zlog':False,
						 'model':'msis','run_model_on_refresh':True}	

		#This will allow us to see what has changed between
		#redraws/refreshes
		self.last_controlstate = self.controlstate.copy() 

		#Init the right-click menu
		self.create_actions()
		self.init_context_menu()




	def changed(self,key):
		"""Was a control described by key 'key' changed since last refresh?"""
		return self.last_controlstate[key]!=self.controlstate[key]

	def plotProperty(self,prop):
		"""Simple convenience function to retrieve a property of the current type of plot"""
		return self.pdh.plottypes[self.pdh.plottype][prop]

	def set_mplparams(self):
		"""Sets default visual appearance of the axes"""
		fs = 5
		ms = 3
		mpl.rcParams['lines.linewidth'] = .6
		mpl.rcParams['lines.markersize'] = ms
		mpl.rcParams['font.size'] = fs

	def reset(self):
		"""Clears the data handler data and the click handler history"""
		self.pdh.clear_data()
		

	def prepare_model_run(self):
		"""Determines which position variables (lat,lon, or alt) are constant,
		given the current settings of the xvar, yvar and zvar. Then reads the 
		approriate values and prepares either flattened gridded input for the 
		ModelRunner or simple 1-d vectors if line plotting"""
		#and should be determined by reading the state of the lineedit widgets

		#Begin by assigning all of the position variables their approprate output
		#from the controls structure. Some of these values will be overwritten
		#since at least one must be on a plot axes if line plot,
		#or at least two if a colored plot (pcolor or contour)
		self.mr.nextrun.vars['Latitude'] = self.controlstate['lat']
		self.mr.nextrun.vars['Longitude'] = self.controlstate['lon']
		self.mr.nextrun.vars['Altitude'] = self.controlstate['alt']
		
		#Now figure out what location variables are varying and which are now fixed
		#(the fixed varibles will be read from the lineedit widgets)
		for var,bnd,npt in (('xvar','xbounds','xnpts'),('yvar','ybounds','ynpts')):
			for posvar in ('Latitude','Longitude','Altitude'):
				#Now we figure out if x variable is Latitude, Longitude, or Altitude,
				#and if it is, for example, Latitude, then we set the latitude for 
				#the pending run to a vector defined by the xbounds, with number of points
				#defined by the xnpts
				if self.controlstate[var]==posvar:
					bounds = self.controlstate[bnd]
					numpts = self.controlstate[npt]
					#Now assign the approriate position variable in the pending model run
					self.mr.nextrun.vars[self.controlstate[var]] = numpy.linspace(bounds[0],bounds[1],numpts)
					self.mr.nextrun.shape = (numpts,1) #The default output shape of a column vector

		#Now we determine from the plottype if we need to grid x and y
		#and reassign the vars in the next run corresponding to the xvar and yvar
		#selection to the newly gridded data
		#Also assign the shape of the expected model output
		if self.pdh.plottypes[self.pdh.plottype]['gridxy']:
			#Fault check
			if self.controlstate['xvar'] not in self.mr.nextrun.vars:
				raise RuntimeError('xvar %s is not a valid position variable!' % (self.controlstate['xvar']))
			if self.controlstate['yvar'] not in self.mr.nextrun.vars:
				raise RuntimeError('yvar %s is not a valid position variable!' % (self.controlstate['yvar']))
			#Make grids
			x = self.mr.nextrun.vars[self.controlstate['xvar']] 
			y = self.mr.nextrun.vars[self.controlstate['yvar']]
			X,Y = numpy.meshgrid(x,y)
			self.mr.nextrun.vars[self.controlstate['xvar']] = X  
			self.mr.nextrun.vars[self.controlstate['yvar']] = Y
			self.mr.nextrun.shape = X.shape #Set the shape, this overrides the default shape
		else: #We do not need to grid data
			#Check that at least one selected variable is a location
			if self.controlstate['xvar'] not in self.mr.nextrun.vars and self.controlstate['yvar'] not in self.mr.nextrun.vars:
				raise RuntimeError('%s and %s are both not valid position variables!' % (self.controlstate['xvar'],self.controlstate['yvar']))

	@QtCore.pyqtSlot()
	def refresh(self,force_full_refresh=False):
		"""Redraws what is on the plot. Trigged on control change"""
		ffr = force_full_refresh

		if self.changed('plottype') or ffr:
			#Determine if we need to rerun the model
			oldplottype = self.pdh.plottypes[self.pdh.plottype]
			newplottype = self.pdh.plottypes[self.controlstate['plottype']]
			if oldplottype['gridxy'] != newplottype['gridxy']: #we are going from vectors to grids or visa-versa
				self.controlstate['run_model_on_refresh']=True #Must force re-run
			self.pdh.plottype=self.controlstate['plottype']

		if self.changed('datetime') or ffr:
			#Force model rerun
			self.controlstate['run_model_on_refresh']=True
			self.mr.nextrun.dt = self.controlstate['datetime']

		if self.changed('lat') or ffr:
			if 'Latitude' not in [self.controlstate['xvar'],self.controlstate['yvar']]:
				#We are holding latitude constant, so we will have to rerun the model
				self.controlstate['run_model_on_refresh'] = True
		
		if self.changed('lon') or ffr:
			if 'Longitude' not in [self.controlstate['xvar'],self.controlstate['yvar']]:
				#We are holding longitude constant, so we will have to rerun the model
				self.controlstate['run_model_on_refresh'] = True

		if self.changed('alt') or ffr:
			if 'Altitude' not in [self.controlstate['xvar'],self.controlstate['yvar']]:
				#We are holding altitude constant, so we will have to rerun the model
				self.controlstate['run_model_on_refresh'] = True
		
		if self.controlstate['run_model_on_refresh'] or ffr:
			self.prepare_model_run()
			self.mr.nextrun() #Trigger next model run
			self.mr() #Trigger storing just created model run as mr.runs[-1]
			self.reset() #Reset the click handler and the plotDataHandler
			#Update the variable bounds in the controlstate
			if self.changed('xvar') or ffr:
				self.controlstate['xbounds'] = self.mr.runs[-1].lims[self.controlstate['xvar']]
			if self.changed('yvar') or ffr:
				self.controlstate['ybounds'] = self.mr.runs[-1].lims[self.controlstate['yvar']]
			if self.changed('zvar') or ffr:
				self.controlstate['zbounds'] = self.mr.runs[-1].lims[self.controlstate['zvar']]

		#Associate data in the data handler based on what variables are desired
		if self.changed('xvar') or self.changed('xbounds') or self.changed('xlog') or self.controlstate['run_model_on_refresh'] or ffr: 
			xdata = self.mr.runs[-1].vars[self.controlstate['xvar']]
			xname = self.controlstate['xvar']
			self.pdh.associate_data('x',xdata,xname,self.controlstate['xbounds'],self.controlstate['xlog'])
			#if self.ch is not None:
			#	self.ch.reset()
		
		if self.changed('yvar') or self.changed('ybounds') or self.changed('ylog') or self.controlstate['run_model_on_refresh'] or ffr: 
			ydata = self.mr.runs[-1].vars[self.controlstate['yvar']]
			yname = self.controlstate['yvar']
			self.pdh.associate_data('y',ydata,yname,self.controlstate['ybounds'],self.controlstate['ylog'])
			#if self.ch is not None:
			#	self.ch.reset()

		if self.changed('zvar') or self.changed('zbounds') or self.changed('zlog') or self.controlstate['run_model_on_refresh'] or ffr:
			zdata = self.mr.runs[-1].vars[self.controlstate['zvar']]
			zname = self.controlstate['zvar']
			self.pdh.associate_data('z',zdata,zname,self.controlstate['zbounds'],self.controlstate['zlog'])
			#if self.ch is not None:
			#	self.ch.reset()

		#Referesh the plots
		self.ax.cla()

		self.pdh.plot()

		#Reset the last controlstate
		self.last_controlstate = self.controlstate.copy() 
		self.canvasrefreshed.emit()

		self.draw()
		
	
	#Menu stuff
	def init_context_menu(self):
		"""Connects a canvas to it's context menu callback"""
		self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu);
	
		self.connect(self,SIGNAL("customContextMenuRequested(QPoint)"),
					self,SLOT("contextMenuRequested(QPoint)"))

	def create_actions(self):
		"""Creates all QT actions which apply to the canvas"""
		self.actions = OrderedDict()
		#Actions which change the scaling of axes
		
		self.actions['xlog'] = QtGui.QAction('&x log-scale', self)
		self.actions['ylog'] = QtGui.QAction('&y log-scale', self)
		self.actions['zlog'] = QtGui.QAction('&color log', self)
		
		#Connect with fancy lambda syntax so I can do it in a loop
		for var in ['x','y','z']:
			self.actions[var+'log'].setCheckable(True) #Make them ticks
			self.connect(self.actions[var+'log'],SIGNAL("toggled(bool)"),
						lambda boo,xyz=var: self.on_logtoggled(boo,xyz))

		#Boundaries of axes
		self.actions['xbounds'] = QtGui.QAction('Set &XLim', self)
		self.actions['ybounds'] = QtGui.QAction('Set &YLim', self)
		self.actions['zbounds'] = QtGui.QAction('Set &CLim', self)
		
		#Connect with fancy lambda syntax so I can do it in a loop
		for var in ['x','y','z']:
			self.connect(self.actions[var+'bounds'],SIGNAL("triggered()"),
						lambda xyz=var: self.on_boundstriggered(xyz))
		
		self.actions['refresh'] = QtGui.QAction('&Refresh', self)
		self.connect(self.actions['refresh'],SIGNAL("triggered()"),self,SLOT('refresh()'))


	#Event handlers
	@QtCore.pyqtSlot('QPoint')
	def contextMenuRequested(self,point):
		menu = QtGui.QMenu()
		sublog = QtGui.QMenu('Log-scale')
		sublim = QtGui.QMenu('Axes limits')
				
		for var in ['x','y']:
			sublog.addAction(self.actions[var+'log'])
			if self.controlstate[var+'log']:
				#Annoying...there's probably a better way to block signals
				self.actions[var+'log'].blockSignals(True)
				self.actions[var+'log'].setChecked(True)
				self.actions[var+'log'].blockSignals(False)

			sublim.addAction(self.actions[var+'bounds'])
		
		if self.plotProperty('gridxy'):
			#If we have a color axes
			sublog.addAction(self.actions['zlog'])
			sublim.addAction(self.actions['zbounds'])
		
		menu.addMenu(sublog)
		menu.addMenu(sublim)
		menu.addAction(self.actions['refresh'])

		# menu._exec is modal, menu.popup is not
		# this means that menu.popup will not stop execution
		menu.exec_(self.mapToGlobal(point))

	@QtCore.pyqtSlot()
	def on_boundstriggered(self,xyz):
		"""Pop up a modal dialog to change the bounds"""
		minflt,maxflt,result = BoundsDialog.getMinMax(initialbounds=self.controlstate[xyz+'bounds'])
		if result:
			if maxflt>minflt:
				print "Changing %s bounds to %.3f %.3f" % (xyz,minflt,maxflt)
				self.controlstate[xyz+'bounds']=(minflt,maxflt)
				print str(self.controlstate[xyz+'bounds'])
			else:
				print "Bad range!"
		else:
			print "User canceled bounds change"
		self.refresh()

	@QtCore.pyqtSlot(bool)
	def on_logtoggled(self,boo,var):
		"""Fancy multi-argument callback (use lambdas in the connect signal call)"""
		print "%s log scale is %s" % (var,str(boo))
		self.controlstate[var.lower()+'log'] = boo

	@QtCore.pyqtSlot('QDateTime')
	def on_datetimechange(self,qdt):
		dt = qdt.toPyDateTime()
		print "Date changed to %s" % (dt.strftime('%c'))
		self.controlstate['datetime']=dt


	@QtCore.pyqtSlot(str)
	def on_xvarchange(self,new_value):
		"""Handles user changing xvar combo box"""
		new_value = str(new_value) #Make sure not QString
		print "X data set in singleMplCanvas to %s" % (new_value)
		self.controlstate['xvar']=new_value
		self.controlstate['xbounds']=self.mr.runs[-1].lims[new_value]

		
	@QtCore.pyqtSlot(str)
	def on_yvarchange(self,new_value):
		"""Handles user changing yvar combo box"""
		new_value = str(new_value) #Make sure not QString
		print "Y data set in singleMplCanvas to %s" % (new_value)
		self.controlstate['yvar']=new_value
		self.controlstate['ybounds']=self.mr.runs[-1].lims[new_value]
		

	@QtCore.pyqtSlot(str)
	def on_zvarchange(self,new_value):
		"""Handles user changing zvar combo box"""
		new_value = str(new_value) #Make sure not QString
		print "Z data set in singleMplCanvas to %s" % (new_value)
		self.controlstate['zvar']=new_value
		self.controlstate['zbounds']=self.mr.runs[-1].lims[new_value]
		

	@QtCore.pyqtSlot(str)
	def on_latlinechange(self,new_value):
		"""Handles user changing single latitude lineedit widget"""
		#Convert string to float
		new_value = str(new_value) #Make sure not QString
		try:
			new_lat = float(new_value)
		except:
			print "Unable to convert entered value %s for latitude to float!" %(new_value)
			return
		#Range check
		if new_lat < -90. or new_lat > 90.:
			print "Invalid value for latitude %f" % (new_lat)
			return
		print "Single latitude changed to %f" %(new_lat)
		self.controlstate['lat'] = new_lat
		

	@QtCore.pyqtSlot(str)
	def on_lonlinechange(self,new_value):
		"""Handles user changing single longitude lineedit widget"""
		#Convert string to float
		new_value = str(new_value) #Make sure not QString
		try:
			new_lon = float(new_value)
		except:
			print "Unable to convert entered value %s for longitude to float!" %(new_value)
			return
		#Range check
		if new_lon > 180.:
			new_lon = new_lon - 360.
		#And again
		if new_lon < -180. or new_lon > 180.:
			print "Invalid value for longitude %f" % (new_lon)
			return
		print "Single longitude changed to %f" %(new_lon)
		self.controlstate['lon'] = new_lon
		

	@QtCore.pyqtSlot(str)
	def on_altlinechange(self,new_value):
		"""Handles user changing single altitude lineedit widget"""
		#Convert string to float
		new_value = str(new_value) #Make sure not QString
		try:
			new_alt = float(new_value)
		except:
			print "Unable to convert entered value %s for altitude to float!" %(new_value)
			return
		#Range check
		if new_alt < 0.:
			print "Invalid value for altitude %f" % (new_alt)
			return
		print "Single altitude changed to %f" %(new_alt)
		self.controlstate['alt'] = new_alt
		

	@QtCore.pyqtSlot(str)
	def on_plottypechange(self,new_value):
		#Sanitize input
		new_value = str(new_value) #Make sure not QString
		if new_value not in self.pdh.plottypes.keys():
			raise ValueError('Invalid plottype set by combobox! %s' %(new_value))
		self.controlstate['plottype'] = new_value
		

class mainCanvas(singleMplCanvas):
	"""The main canvas for plotting the main visualization"""
	def __init__(self, *args, **kwargs):
		singleMplCanvas.__init__(self, *args, **kwargs)
		self.refresh(force_full_refresh=True) #This will set everything up to plot as per current controlstate
		
	def update_figure(self):
		pass
	
class auxCanvas(singleMplCanvas):
	"""The secondary canvas that floats above the controls"""
	def __init__(self, *args, **kwargs):
		singleMplCanvas.__init__(self, *args, **kwargs)
		self.controlstate['plottype']='line'
		self.refresh(force_full_refresh=True) #This will set everything up to plot as per current controlstate
		
	def update_figure(self):
		pass

class AtModExplorerApplicationWindow(QtGui.QMainWindow):
	import textwrap
	@QtCore.pyqtSlot()
	def update_controls(self):
		"""Called when the main canvas refreses"""
		print "update_controls called"
		f107 = str(self.maincanv.mr.runs[-1].drivers['f107'])
		f107a = str(self.maincanv.mr.runs[-1].drivers['f107a'])
		ap = str(self.maincanv.mr.runs[-1].drivers['ap'][0])
		self.tbox.setText('Activity From IRI Database:\n%s\nF-10.7:\n  %s\nF10.7 81-day Average:\n  %s\nAP:\n  %s' % (self.maincanv.controlstate['datetime'].strftime('%c'),
			f107,f107a,ap))

	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		self.setWindowTitle("AtModExplorer")

		self.file_menu = QtGui.QMenu('&File', self)
		self.file_menu.addAction('&Quit', self.fileQuit,
								 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
		self.menuBar().addMenu(self.file_menu)

		self.help_menu = QtGui.QMenu('&Help', self)
		self.menuBar().addSeparator()
		self.menuBar().addMenu(self.help_menu)

		self.help_menu.addAction('&About', self.about)

		self.main_widget = QtGui.QWidget(self)


		#Make our canvases
		self.maincanv = mainCanvas(self.main_widget,appwindow=self,figsize=(6,5), dpi=200)
		
		self.auxcanv = auxCanvas(self.main_widget,appwindow=self,figsize=(1.5,1.5), dpi=200)
		

		#Attach the clicked event
		self.maincanv.mpl_connect('button_press_event', self.canvas_clicked)

		QtCore.QObject.connect(self.maincanv,SIGNAL('canvasrefreshed()'),self,SLOT('update_controls()'))

		#self.mpl_toolbar = NavigationToolbar(self.maincanv,self.main_widget)

		self.auxcanv.mpl_connect('button_press_event', self.canvas_clicked)
		#Create the click property
		self.last_click = None

		#Make our layouts (main is vertical, containing two horizontal, top horizontal contains two vertical)
		vboxmain = QtGui.QVBoxLayout(self.main_widget) # Main box
		hboxtop = QtGui.QHBoxLayout() # Container for Main Canvas and Control Panel
		vboxtopl = QtGui.QVBoxLayout() # Control Panel and Aux Canvas
		hboxtoplloc = QtGui.QHBoxLayout() #For ephemeris fields
		hboxtoplhms = QtGui.QHBoxLayout() #For Hour, Minute, Second

		vboxtopr = QtGui.QVBoxLayout()	# Main Canvas
		#vboxtopr.addWidget(self.mpl_toolbar)
		vboxtopr.addWidget(self.maincanv)

		hboxbot = QtGui.QHBoxLayout() # Bottom Row of Buttons / Text inputs
		
	

		#Create the auxillary canvas
		#self.auxcanv = auxCanvas(self.main_widget,appwindow=self,figsize=(2,1.5), dpi=200)
		#vboxtopl.addWidget(self.auxcanv)
		vboxtopl.addWidget(self.auxcanv)
		vboxtopl.addStretch(1)

		#Create the output box
		self.tbox = QtGui.QTextEdit('MSIS output')
		vboxtopl.addWidget(self.tbox)
		self.update_controls() #Init the textbox

		#Create the plot button
		pbutton = QtGui.QPushButton('Plot')
		pbutton.clicked[bool].connect(self.plot_clicked)
		vboxtopl.addWidget(pbutton)

		
		#Create the time changer
		tlabel = QtGui.QLabel()
		tlabel.setText('UT Time')
		self.tdte = QtGui.QDateTimeEdit()
		self.tdte.setDateTime(self.maincanv.controlstate['datetime'])
		self.tdte.dateTimeChanged['QDateTime'].connect(self.maincanv.on_datetimechange)
		vboxtopl.addWidget(tlabel)
		vboxtopl.addWidget(self.tdte)


		loclabel = QtGui.QLabel()
		loclabel.setText('Geodetic Latitude, Longitude, Altitude(km)')
		#Create the fill in text boxes for latitude, longitude and altitude
		self.qlelat = QtGui.QLineEdit()		
		self.qlelon = QtGui.QLineEdit()
		self.qlealt = QtGui.QLineEdit()
		#Set intial values to intial main controlstate
		self.set_locations(self.maincanv.controlstate)

		#Connect to maincanv events
		self.qlelat.textChanged[str].connect(self.maincanv.on_latlinechange)
		self.qlelon.textChanged[str].connect(self.maincanv.on_lonlinechange)
		self.qlealt.textChanged[str].connect(self.maincanv.on_altlinechange)

		#Connect to auxcanv events
		self.qlelat.textChanged[str].connect(self.auxcanv.on_latlinechange)
		self.qlelon.textChanged[str].connect(self.auxcanv.on_lonlinechange)
		self.qlealt.textChanged[str].connect(self.auxcanv.on_altlinechange)

		#Add the location controls to the GUI
		vboxtopl.addWidget(loclabel)
		hboxtoplloc.addWidget(self.qlelat)
		hboxtoplloc.addWidget(self.qlelon)
		hboxtoplloc.addWidget(self.qlealt)
		vboxtopl.addLayout(hboxtoplloc)

		#Create the variable selectors for the main window
		self.comboModel = QtGui.QComboBox()
		self.comboModel.addItem('NRLMSISE00')
		self.comboModelLab = QtGui.QLabel('Model')
		comboModelVbox = QtGui.QVBoxLayout()
		comboModelVbox.addWidget(self.comboModelLab)
		comboModelVbox.addWidget(self.comboModel)
		comboModelVbox.addStretch(1)
		
		self.comboXvar = QtGui.QComboBox()
		self.comboXvaraux = QtGui.QComboBox()
		self.comboXlab = QtGui.QLabel('X Var')
		comboXVbox = QtGui.QVBoxLayout()
		comboXVbox.addWidget(self.comboXlab)
		comboXVbox.addWidget(self.comboXvar)
		comboXVbox.addWidget(self.comboXvaraux)
		

		self.comboYvar = QtGui.QComboBox()
		self.comboYvaraux = QtGui.QComboBox()
		self.comboYlab = QtGui.QLabel('Y Var')
		comboYVbox = QtGui.QVBoxLayout()
		comboYVbox.addWidget(self.comboYlab)
		comboYVbox.addWidget(self.comboYvar)
		comboYVbox.addWidget(self.comboYvaraux)

		self.comboZvar = QtGui.QComboBox()
		self.comboZlab = QtGui.QLabel('Color Var')
		comboZVbox = QtGui.QVBoxLayout()
		comboZVbox.addWidget(self.comboZlab)
		comboZVbox.addWidget(self.comboZvar)
		comboZVbox.addStretch(1)

		self.comboPlotType = QtGui.QComboBox()
		self.comboPlotTypeLab = QtGui.QLabel('Plot Type')
		comboPlotTypeVbox = QtGui.QVBoxLayout()
		comboPlotTypeVbox.addWidget(self.comboPlotTypeLab)
		comboPlotTypeVbox.addWidget(self.comboPlotType)
		comboPlotTypeVbox.addStretch(1)

		self.comboPlotType.addItems(self.maincanv.pdh.plottypes.keys()) #adds all plot types defined in plotDataHandler
		self.comboPlotType.activated[str].connect(self.maincanv.on_plottypechange)
		
		for cbox in [self.comboXvar,self.comboYvar,self.comboZvar]:
			cbox.addItems(self.maincanv.mr.runs[-1].vars.keys()) #adds all possible variables from MSIS runs
		
		for cbox in [self.comboXvaraux,self.comboYvaraux]:
			cbox.addItems(self.auxcanv.mr.runs[-1].vars.keys()) #adds all possible variables from MSIS runs

		#Set the current value for the comboboxes
		self.set_comboboxes(self.maincanv.controlstate)
		self.set_aux_comboboxes(self.auxcanv.controlstate)

		#Now bind change methods to comboboxes
		#Note this is the 'right'
		#way of doing this...but I don't like it's verbosity
		#QtCore.QObject.connect(comboXvar, QtCore.SIGNAL('currentIndexChanged(const QString&)), self.onClicked)
		
		self.comboXvar.activated[str].connect(self.maincanv.on_xvarchange)
		self.comboXvaraux.activated[str].connect(self.auxcanv.on_xvarchange)
				
		self.comboYvar.activated[str].connect(self.maincanv.on_yvarchange)
		self.comboYvaraux.activated[str].connect(self.auxcanv.on_yvarchange)
		
		self.comboZvar.activated[str].connect(self.maincanv.on_zvarchange)
		

		#Add to the bottom row widget
		hboxbot.addLayout(comboModelVbox)
		hboxbot.addLayout(comboXVbox)
		hboxbot.addLayout(comboYVbox)
		hboxbot.addLayout(comboZVbox)
		hboxbot.addLayout(comboPlotTypeVbox)

		#Put together the layouts
		hboxtop.addLayout(vboxtopl)
		hboxtop.addLayout(vboxtopr)
		vboxmain.addLayout(hboxtop)
		vboxmain.addLayout(hboxbot)

		#Left panel

		self.main_widget.setFocus()
		self.setCentralWidget(self.main_widget)

		self.statusBar().showMessage("All hail matplotlib!", 2000)

		
	def plot_clicked(self,tf):
		self.maincanv.refresh()
		self.auxcanv.refresh()

	def canvas_clicked(self,event):
		#Ignore everything except left-clicks
		if event.button!=1:
			return

		ax = event.inaxes #Which axes the click occured in
		if self.last_click is not None:
			for ln in self.last_click:
				ln.set_visible(False)

		if ax is not None: #If the click occured in a axes
			canv = ax.figure.canvas
			
			#Click location
			x = event.xdata 
			y = event.ydata 
			if canv.controlstate['plottype']=='map':
				x,y = canv.pdh.map(x,y,inverse=True)
			
			#Associated data 
			xdata = canv.pdh.x.flatten()
			ydata = canv.pdh.y.flatten()
			zdata = canv.pdh.z.flatten()

			#Figure out which point was closest to this click
			
			#Compute the Euclidian distance between all points in the userdata and the clicked one
			dist = np.sqrt((xdata-x)**2+(ydata-y)**2)

			ind = dist.argmin() #index of the closest point to the click location 
			
			clx = xdata[ind]#Closest datapoint x coordinate 
			cly = ydata[ind]#Closest datapoint y coordinate 
				
			if canv.controlstate['plottype']=='map': 
				#Plot point at clicked location and add resulting lines.line2D to the 'clicked' list
				#Add the plotted line to the clicked list
				self.last_click = canv.pdh.map.plot(clx,cly,'mo',alpha=.8,latlon=True)
				self.auxcanv.controlstate['lat']=numpy.floor(clx)
				self.auxcanv.controlstate['lon']=numpy.floor(cly)
				self.auxcanv.controlstate['xvar']='Altitude'
				self.auxcanv.controlstate['yvar']=self.maincanv.controlstate['zvar']
				self.set_locations(self.auxcanv.controlstate)
				self.set_aux_comboboxes(self.auxcanv.controlstate)
				self.auxcanv.refresh()

			else:
				#Plot point at clicked location and add resulting lines.line2D to the 'clicked' list
				#Add the plotted line to the clicked list
				self.last_click = ax.plot(clx,cly,'mo',alpha=.8)
			
			if canv.pdh.plottypes[canv.pdh.plottype]['gridxy']:
				clz = zdata[ind]#Closest datapoint z coordinate 
				#Show a message for each click in the app status bar
				self.statusBar().showMessage("%s:%f, %s:%f, %s:%f" % (canv.controlstate['xvar'],clx,
																	  canv.controlstate['yvar'],cly,
																	  canv.controlstate['zvar'],clz), 2000)

			else:
				self.statusBar().showMessage("%s:%f, %s:%f" % (canv.controlstate['xvar'],clx,
															   canv.controlstate['yvar'],cly), 2000)
				clz = numpy.nan
				
			#Redraw
			canv.draw()
			

	def set_locations(self,controlstate):
		"""Sets all location controls to match the controlstate"""
		self.qlelat.setText(str(controlstate['lat']))
		self.qlelon.setText(str(controlstate['lon']))
		self.qlealt.setText(str(controlstate['alt']))

	def set_comboboxes(self,controlstate):
		"""Sets the X, Y, and Z Variable Choosers By reading a canvas controlstate dict"""
		indX = self.comboXvar.findText(controlstate['xvar'])
		indY = self.comboYvar.findText(controlstate['yvar'])
		indZ = self.comboZvar.findText(controlstate['zvar'])
		self.comboXvar.setCurrentIndex(indX)
		self.comboYvar.setCurrentIndex(indY)
		self.comboZvar.setCurrentIndex(indZ)

	def set_aux_comboboxes(self,controlstate):
		#Init auxillary X and Y to be same as main Y and Z
		indX = self.comboXvaraux.findText(controlstate['xvar'])
		indY = self.comboYvaraux.findText(controlstate['yvar'])
		self.comboXvaraux.setCurrentIndex(indX)
		self.comboYvaraux.setCurrentIndex(indY)
		
		
	def fileQuit(self):
		self.close()

	def closeEvent(self, ce):
		self.fileQuit()

	def about(self):
		QtGui.QMessageBox.about(self, "About",
			textwrap.dedent("""The AtModExplorer (C) 2014 University of Colorado Space Environemnt Data Analysis Group
								A tool for interactive viewing of output from emperical models of the atmosphere.
								For education use only. Written by Liam Kilcommons <liam.kilcommons@colorado.edu>
			"""))


