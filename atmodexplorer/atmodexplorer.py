import sys
import os
import random
from matplotlib.backends import backend_qt4
import matplotlib.widgets as widgets
import matplotlib.axes
from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import SIGNAL,SLOT,pyqtSlot,pyqtSignal

#Main imports
import numpy as np

#import pandas as pd
import sys, pdb, textwrap
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
import logging
logging.basicConfig(level=logging.INFO)

from atmodbackend import *
#def lim_pad(lims):
#	"""Adds a little extra to the limits to make plots more visible"""
#	delta = lims[1]-lims[0]
#	new_lims = (lims[0]-delta*.1,lims[1]+delta*.1)
#	print lims, new_lims
#	return new_lims

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

	# static method to create the dialog and return (axmin,axmax,True/False)
	@staticmethod
	def getMinMax(parent = None,initialbounds=None):
		dialog = BoundsDialog(parent,initialbounds=initialbounds)
		result = dialog.exec_()
		minflt,maxflt = dialog.getfloats()
		return (minflt,maxflt,result == QtGui.QDialog.Accepted)

class singleMplCanvas(FigureCanvas):
	"""This is also ultimately a QWidget and a FigureCanvasAgg"""
	#Prepare the signals that this can emit
	#this signal tells everything attached that the canvas has been redrawn
	#it doesn't carry any data, because the updated model run is in
	#the attached plotDataHandler (at self.pdh.runs[-1])
	#and the new state of any control (UI control or internal) that has changed is in the 
	#control state dictionary (self.controlstate)
	canvasrefreshed = pyqtSignal()

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
		self.ax = self.fig.add_subplot(111)
		#Create a data handler and a click handler
		self.pdh = PlotDataHandler(self)
		
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
						 'model':'msis','run_model_on_refresh':True,'differencemode':False,\
						 'xmulti':False,'ymulti':False}	

		#This will allow us to see what has changed between
		#redraws/refreshes
		self.last_controlstate = self.controlstate.copy() 

		#Init the right-click menu
		self.create_actions()
		self.init_context_menu()

	def changed(self,key=None):
		"""Was a control described by key 'key' changed since last refresh? If no key, return dictionary of all changed items"""
		if key is not None:
			return self.last_controlstate[key]!=self.controlstate[key]
		else: #return a dictionary of all changed controls
			changed_items = dict()
			for kee in self.controlstate:
				if self.changed(key=kee):
					changed_items[kee]=self.controlstate[kee]
			return changed_items

	def plotProperty(self,prop):
		"""Simple convenience function to retrieve a property of the current type of plot"""
		return self.pdh.plottypes[self.pdh.plottype][prop]

	def set_mplparams(self):
		"""Sets default visual appearance of the axes"""
		fs = 5
		ms = 3
		mpl.rcParams['lines.linewidth'] = .5
		mpl.rcParams['lines.markersize'] = ms
		mpl.rcParams['font.size'] = fs

	def reset(self):
		"""Clears the data handler data"""
		self.pdh.clear_data()

	def is_multi(self,coord):
		"""Convenience function for testing whether the currently selected x or y variables are multiple vars on same axes"""
		return hasattr(self.controlstate[coord+'var'],'__iter__') #just tests if the controlstate is a list/tuple
			
	def is_position(self,coord):
		"""Convenience function for testing whether the currently selected x or y variables are positions"""
		if not self.is_multi(coord):
			return self.controlstate[coord+'var'] in self.mr.nextrun.vars
		else:
			return any(v in self.mr.nextrun.vars for v in self.controlstate[coord+'var'])  
		
	def prepare_model_run(self):
		"""Determines which position variables (lat,lon, or alt) are constant,
		given the current settings of the xvar, yvar and zvar. Then reads the 
		approriate values and prepares either flattened gridded input for the 
		ModelRunner or simple 1-d vectors if line plotting"""
	
		#Begin by assigning all of the position variables their approprate output
		#from the controls structure. These are all single (scalar) values set by
		#the QLineEdit widgets for Lat, Lon and Alt.
		#Everything is GEODETIC, not GEOCENTRIC, because that's what MSIS expects.
		#Some of these values will be overwritten
		#since at least one must be on a plot axes if line plot,
		#or at least two if a colored plot (pcolor or contour)
		self.mr.nextrun.vars['Latitude'] = self.controlstate['lat']
		self.mr.nextrun.vars['Longitude'] = self.controlstate['lon']
		self.mr.nextrun.vars['Altitude'] = self.controlstate['alt']

		#Copy out the drivers from the controlstate (only copy those that are exposed via the model's __init__)
		self.mr.nextrun.drivers['dt'] = self.controlstate['datetime']

		#Now we determine from the plottype if we need to grid x and y
		
		if self.plotProperty('gridxy'):
			#Fault checks
			if not self.is_position('x'): #vars dict starts only with position and time
				raise RuntimeError('xvar %s is not a valid position variable!' % (self.controlstate['xvar']))
			else:
				#self.mr.nextrun.lims[self.controlstate['xvar']] = self.controlstate['xbounds']
				self.mr.nextrun.npts[self.controlstate['xvar']] = self.controlstate['xnpts']
				self.mr.nextrun.set_x(self.controlstate['xvar'])

			if not self.is_position('y'):
				raise RuntimeError('yvar %s is not a valid position variable!' % (self.controlstate['yvar']))
			else:
				#self.mr.nextrun.lims[self.controlstate['yvar']] = self.controlstate['ybounds']
				self.mr.nextrun.npts[self.controlstate['yvar']] = self.controlstate['ynpts']
				self.mr.nextrun.set_y(self.controlstate['yvar'])
			
		else: #We do not need to grid data
			#Check that at least one selected variable is a location
			#Handle multiple variables on an axis
			if self.is_multi('x') and self.is_position('x'):
				self.controlstate['xvar']=self.controlstate['xvar'][0]
				raise RuntimeError('Multiple plotting of position variables is not allowed!')

			elif self.is_multi('y') and self.is_position('y'):
				self.controlstate['yvar']=self.controlstate['yvar'][0]
				raise RuntimeError('Multiple plotting of position variables is not allowed!')
				
			elif not self.is_position('x') and not self.is_position('y'):
				raise RuntimeError('%s and %s are both not valid position variables!' % (self.controlstate['xvar'],self.controlstate['yvar']))
			
			elif not self.is_multi('x') and self.is_position('x'): #It's scalar, so check if it's a position
				#self.mr.nextrun.lims[self.controlstate['xvar']] = self.controlstate['xbounds']
				self.mr.nextrun.npts[self.controlstate['xvar']] = self.controlstate['xnpts']
				self.mr.nextrun.set_x(self.controlstate['xvar'])

			elif not self.is_multi('y') and self.is_position('y'): #It's scalar, so check if it's a position
				self.mr.nextrun.npts[self.controlstate['yvar']] = self.controlstate['ynpts']
				self.mr.nextrun.set_y(self.controlstate['yvar'])
			else:
				raise RuntimeError('Nonsensical variables: xvar:%s\n yvar:%s\n' % (repr(self.controlstate['xvar']),repr(self.controlstate['yvar'])))


	@pyqtSlot()
	def refresh(self,force_full_refresh=False, force_autoscale=False):
		"""Redraws what is on the plot. Trigged on control change"""
		ffr = force_full_refresh
		fauto = force_autoscale

		if self.changed('plottype') or ffr:
			#Determine if we need to rerun the model
			oldplottype = self.pdh.plottypes[self.pdh.plottype]
			newplottype = self.pdh.plottypes[self.controlstate['plottype']]
			if oldplottype['gridxy'] != newplottype['gridxy']: #we are going from vectors to grids or visa-versa
				self.controlstate['run_model_on_refresh']=True #Must force re-run
			self.pdh.plottype=self.controlstate['plottype']
			#Also reset any multiplotting
			self.controlstate['xmulti']=False
			self.controlstate['ymulti']=False

		if self.changed('datetime') or ffr:
			#Force model rerun
			self.controlstate['run_model_on_refresh']=True
			self.mr.nextrun.dt = self.controlstate['datetime']

		if self.changed('lat') or ffr:
			if 'Latitude' not in [self.controlstate['xvar'],self.controlstate['yvar']]:
				self.mr.nextrun.hold_constant('Latitude')
				#We are holding latitude constant, so we will have to rerun the model
				self.controlstate['run_model_on_refresh'] = True
		
		if self.changed('lon') or ffr:
			if 'Longitude' not in [self.controlstate['xvar'],self.controlstate['yvar']]:
				self.mr.nextrun.hold_constant('Longitude')
				#We are holding longitude constant, so we will have to rerun the model
				self.controlstate['run_model_on_refresh'] = True

		if self.changed('alt') or ffr:
			if 'Altitude' not in [self.controlstate['xvar'],self.controlstate['yvar']]:
				self.mr.nextrun.hold_constant('Altitude')
				#We are holding altitude constant, so we will have to rerun the model
				self.controlstate['run_model_on_refresh'] = True
		
		if self.changed('differencemode'):
			self.controlstate['run_model_on_refresh'] = True
		
		#Reset multi plotting if we've changed the plot type
		if self.changed('plottype') or ffr:
			pass

		if self.controlstate['run_model_on_refresh'] or ffr:
			self.prepare_model_run()
			self.mr.nextrun.populate() #Trigger next model run
			self.mr() #Trigger storing just created model run as mr.runs[-1]
			self.reset() #Reset the click handler and the plotDataHandler
			
		#Always grab the most current data	
		xdata,xlims = self.mr[self.controlstate['xvar']] #returns data,lims
		ydata,ylims = self.mr[self.controlstate['yvar']] #returns data,lims
		zdata,zlims = self.mr[self.controlstate['zvar']] #returns data,lims
		
		#print '%s:%s' % (self.controlstate['xvar'],repr(xlims))
		#print '%s:%s' % (self.controlstate['yvar'],repr(ylims))
		#print '%s:%s' % (self.controlstate['zvar'],repr(zlims))
		
		#Reset the bounds, multiplotting and turn of log scaling if we have changed any variables or switched on or off of difference mode
		if (self.changed('xvar') and not self.controlstate['xmulti']) or \
			(self.changed('yvar') and not self.controlstate['ymulti']) or \
			 self.changed('zvar') or self.changed('differencemode') or ffr:
			self.controlstate['xbounds'] = xlims
			self.controlstate['ybounds'] = ylims
			self.controlstate['zbounds'] = zlims
			self.controlstate['xlog']=False
			self.controlstate['ylog']=False
			self.controlstate['zlog']=False
			self.controlstate['xmulti']=False
			self.controlstate['ymulti']=False

		if fauto: #Only rescale
			self.controlstate['xbounds'] = xlims
			self.controlstate['ybounds'] = ylims
			self.controlstate['zbounds'] = zlims
			
		#Associate data in the data handler based on what variables are desired
		if self.changed('xvar') or self.changed('xbounds') or self.changed('xlog') or self.controlstate['run_model_on_refresh'] or ffr: 
			xname = self.controlstate['xvar']
			self.pdh.associate_data('x',xdata,xname,self.controlstate['xbounds'],self.controlstate['xlog'],multi=self.controlstate['xmulti'])
			
		if self.changed('yvar') or self.changed('ybounds') or self.changed('ylog') or self.controlstate['run_model_on_refresh'] or ffr: 
			yname = self.controlstate['yvar']
			self.pdh.associate_data('y',ydata,yname,self.controlstate['ybounds'],self.controlstate['ylog'],multi=self.controlstate['ymulti'])
			
		if self.changed('zvar') or self.changed('zbounds') or self.changed('zlog') or self.controlstate['run_model_on_refresh'] or ffr:
			zname = self.controlstate['zvar']
			self.pdh.associate_data('z',zdata,zname,self.controlstate['zbounds'],self.controlstate['zlog'])
			
		#Referesh the plots
		self.ax.cla()

		self.pdh.plot()

		#Reset the last controlstate
		self.last_controlstate = self.controlstate.copy() 

		#Emit the signal to QT that the canvas has refreshed
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
		self.connect(self.actions['refresh'],SIGNAL("triggered()"),lambda tf=True: self.refresh(force_full_refresh=tf))

		self.actions['statistics'] = QtGui.QAction('&Statistics', self)
		self.connect(self.actions['statistics'],SIGNAL("triggered()"),self,SLOT('statisticsRequested()'))

		self.actions['difference'] = QtGui.QAction('&Difference Mode', self)
		self.actions['difference'].setCheckable(True)
		self.connect(self.actions['difference'],SIGNAL("toggled(bool)"),self,SLOT('on_differencemodetoggled(bool)'))

	def make_multi(self,coord,var):
			self.controlstate[coord+'multi']=True
			if not hasattr(self.controlstate[coord+'var'],'__iter__'):
				self.controlstate[coord+'var'] = [self.controlstate[coord+'var'],var]
			else:
				self.controlstate[coord+'var'].append(var)
			logging.info('Now setting %s to multi for variable %s' % (coord,var))
			self.refresh(force_autoscale=True)

	#Event handlers
	@pyqtSlot('QPoint')
	def contextMenuRequested(self,point):
		menu = QtGui.QMenu()
		sublog = QtGui.QMenu('Log-scale')
		sublim = QtGui.QMenu('Axes limits')
				
		for var in ['x','y']:
			#Only non-position variables can be log
			if not self.is_position(var):
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
		
		#Only build overplots for non-position axes
		if self.plotProperty('overplot_ready'):
			if not self.is_position('x'):
				suboverx = QtGui.QMenu('Overplot on X')
				for v in self.mr.runs[-1].vars:
					action = QtGui.QAction('%s' % (v), self)
					self.connect(action,SIGNAL("triggered()"),lambda var=v: self.make_multi('x',var))
					suboverx.addAction(action)
				menu.addMenu(suboverx)
			
			if not self.is_position('y'):
				subovery = QtGui.QMenu('Overplot on Y')
				for v in self.mr.runs[-1].vars:
					action = QtGui.QAction('%s' % (v), self)
					self.connect(action,SIGNAL("triggered()"),lambda var=v: self.make_multi('y',var))
					subovery.addAction(action)
				menu.addMenu(subovery)

		menu.addMenu(sublog)
		menu.addMenu(sublim)
		menu.addAction(self.actions['refresh'])
		menu.addAction(self.actions['statistics'])
		menu.addAction(self.actions['difference'])

		# menu._exec is modal, menu.popup is not
		# this means that menu.popup will not stop execution
		menu.exec_(self.mapToGlobal(point))

	@pyqtSlot()
	def statisticsRequested(self):
		#for now, just display in the tbox
		mystr = ''
		for key in self.pdh.statistics:
			try:
				if np.abs(self.pdh.statistics[key]) < 10000. and np.abs(self.pdh.statistics[key])>.0001:
					mystr+='%s:%.3f\n' % (key,self.pdh.statistics[key])
				else:
					mystr+='%s:%.3e\n' % (key,self.pdh.statistics[key])
			except:
				mystr+='%s:%s\n' % (key,repr(self.pdh.statistics[key]))

		self.aw.tbox.setText(mystr)

	@pyqtSlot(bool)
	def on_differencemodetoggled(self,boo):
		"""Catches signals from difference mode checkbox"""
		self.mr.differencemode = boo
		self.controlstate['differencemode']=boo

	@pyqtSlot()
	def on_boundstriggered(self,xyz):
		"""Pop up a modal dialog to change the bounds"""
		minflt,maxflt,result = BoundsDialog.getMinMax(initialbounds=self.controlstate[xyz+'bounds'])
		if result:
			if maxflt>minflt:
				logging.info( "Changing %s bounds to %.3f %.3f" % (xyz,minflt,maxflt))
				self.controlstate[xyz+'bounds']=(minflt,maxflt)
				logging.info( str(self.controlstate[xyz+'bounds']))
			else:
				logging.warn( "Bad range!")
		else:
			logging.info( "User canceled bounds change" )
		self.refresh()

	@pyqtSlot(bool)
	def on_logtoggled(self,boo,var):
		"""Fancy multi-argument callback (use lambdas in the connect signal call)"""
		logging.info( "%s log scale is %s" % (var,str(boo)))
		self.controlstate[var.lower()+'log'] = boo
		self.refresh()

	@pyqtSlot('QDateTime')
	def on_datetimechange(self,qdt):
		dt = qdt.toPyDateTime()
		logging.info( "Date changed to %s" % (dt.strftime('%c')))
		self.controlstate['datetime']=dt


	@pyqtSlot(str)
	def on_xvarchange(self,new_value):
		"""Handles user changing xvar combo box"""
		new_value = str(new_value) #Make sure not QString
		logging.info( "X data set in singleMplCanvas to %s" % (new_value))
		self.controlstate['xvar']=new_value
		self.controlstate['xbounds']=self.mr.runs[-1].lims[new_value]

		
	@pyqtSlot(str)
	def on_yvarchange(self,new_value):
		"""Handles user changing yvar combo box"""
		new_value = str(new_value) #Make sure not QString
		logging.info( "Y data set in singleMplCanvas to %s" % (new_value))
		self.controlstate['yvar']=new_value
		self.controlstate['ybounds']=self.mr.runs[-1].lims[new_value]
		

	@pyqtSlot(str)
	def on_zvarchange(self,new_value):
		"""Handles user changing zvar combo box"""
		new_value = str(new_value) #Make sure not QString
		logging.info( "Z data set in singleMplCanvas to %s" % (new_value))
		self.controlstate['zvar']=new_value
		self.controlstate['zbounds']=self.mr.runs[-1].lims[new_value]
		

	@pyqtSlot(str)
	def on_latlinechange(self,new_value):
		"""Handles user changing single latitude lineedit widget"""
		#Convert string to float
		new_value = str(new_value) #Make sure not QString
		try:
			new_lat = float(new_value)
		except:
			logging.warn( "Unable to convert entered value %s for latitude to float!" %(new_value))
			return
		#Range check
		if new_lat < -90. or new_lat > 90.:
			logging.info( "Invalid value for latitude %f" % (new_lat))
			return
		logging.info( "Single latitude changed to %f" %(new_lat))
		self.controlstate['lat'] = new_lat
		

	@pyqtSlot(str)
	def on_lonlinechange(self,new_value):
		"""Handles user changing single longitude lineedit widget"""
		#Convert string to float
		new_value = str(new_value) #Make sure not QString
		try:
			new_lon = float(new_value)
		except:
			logging.warn( "Unable to convert entered value %s for longitude to float!" %(new_value))
			return
		#Range check
		if new_lon > 180.:
			new_lon = new_lon - 360.
		#And again
		if new_lon < -180. or new_lon > 180.:
			logging.warn( "Invalid value for longitude %f" % (new_lon))
			return
		logging.info( "Single longitude changed to %f" %(new_lon))
		self.controlstate['lon'] = new_lon
		

	@pyqtSlot(str)
	def on_altlinechange(self,new_value):
		"""Handles user changing single altitude lineedit widget"""
		#Convert string to float
		new_value = str(new_value) #Make sure not QString
		try:
			new_alt = float(new_value)
		except:
			logging.warn( "Unable to convert entered value %s for altitude to float!" %(new_value))
			return
		#Range check
		if new_alt < 0.:
			logging.warn( "Invalid value for altitude %f" % (new_alt))
			return
		logging.info( "Single altitude changed to %f" %(new_alt))
		self.controlstate['alt'] = new_alt
		

	@pyqtSlot(str)
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
		#self.ax.set_position([0.1,.3,.7,.5])
		self.refresh(force_full_refresh=True) #This will set everything up to plot as per current controlstate
		
	def apply_lipstick(self):
		"""Called on each replot, allows cosmetic adjustment"""
		#self.fig.subplots_adjust(left=0.05,bottom=0.05,top=.95,right=.95)
		fs = 5
		w = .5
		lw = .3
		lp = 0
		pd = .5
		if self.pdh.plottype=='pcolor':
			

			mpl.artist.setp(self.ax.get_xmajorticklabels(),size=fs,rotation=30)
			mpl.artist.setp(self.ax.get_ymajorticklabels(),size=fs)
			mpl.artist.setp(self.pdh.cb.ax.get_xmajorticklabels(),size=fs,rotation=45)
						
			#Label is a text object
			self.ax.xaxis.label.set_fontsize(fs)
			self.ax.yaxis.label.set_fontsize(fs)
			self.ax.xaxis.labelpad=lp
			self.ax.yaxis.labelpad=lp
			
			self.ax.title.set_fontsize(fs)
			self.ax.title.set_fontweight('bold')
			
			#Adjust tick size
			self.ax.xaxis.set_tick_params(width=w,pad=pd)
			self.ax.yaxis.set_tick_params(width=w,pad=pd)

			#Colorbar Ticks
			self.pdh.cb.ax.xaxis.set_tick_params(width=w,pad=pd+.5)
			self.pdh.cb.ax.yaxis.set_tick_params(width=w,pad=pd+.5)
			self.pdh.cb.outline.set_linewidth(w)

			self.ax.grid(True,linewidth=.1)
			#Adjust axes border size
			for axis in ['top','bottom','left','right']:
	  			self.ax.spines[axis].set_linewidth(lw)
	  			#self.pdh.cb.spines[axis].set_linewidth(lw)
	  				
		elif self.pdh.plottype=='map':
			#Colorbar Ticks

			mpl.artist.setp(self.pdh.cb.ax.get_xmajorticklabels(),size=fs,rotation=45)
			self.pdh.cb.ax.xaxis.set_tick_params(width=w,pad=pd+.5)
			self.pdh.cb.ax.yaxis.set_tick_params(width=w,pad=pd+.5)
			self.pdh.cb.outline.set_linewidth(w)
						#Adjust axes border size
			for axis in ['top','bottom','left','right']:
	  			self.ax.spines[axis].set_linewidth(lw)
	  			#self.pdh.cb.spines[axis].set_linewidth(lw)

class auxCanvas(singleMplCanvas):
	"""The secondary canvas that floats above the controls"""
	def __init__(self, *args, **kwargs):
		singleMplCanvas.__init__(self, *args, **kwargs)
		#self.ax.set_position([0.25,0.15,0.7,0.7])
		self.controlstate['plottype']='line'
		self.refresh(force_full_refresh=True) #This will set everything up to plot as per current controlstate
		
	def apply_lipstick(self):
		"""Called on each replot, allows cosmetic adjustment"""
		self.fig.subplots_adjust(left=.15,bottom=.15)
		fs = 4
		w = .5
		lw = .5
		lp = 0
		pd = .5

		mpl.artist.setp(self.ax.get_xmajorticklabels(),size=fs,rotation=30)
		mpl.artist.setp(self.ax.get_ymajorticklabels(),size=fs)
		
		#Label is a text object
		self.ax.xaxis.label.set_fontsize(fs)
		self.ax.yaxis.label.set_fontsize(fs)
		self.ax.xaxis.labelpad=lp
		self.ax.yaxis.labelpad=lp
		
		self.ax.title.set_fontsize(fs+1)

		#Adjust tick size
		self.ax.xaxis.set_tick_params(width=w,pad=pd)
		self.ax.yaxis.set_tick_params(width=w,pad=pd)
		#Adjust axes border size
		for axis in ['top','bottom','left','right']:
  			self.ax.spines[axis].set_linewidth(lw)
		#self.figure.tight_layout()
		
class AtModExplorerApplicationWindow(QtGui.QMainWindow):
	import textwrap
	@pyqtSlot()
	def update_controls(self):
		"""Called when the main canvas refreses"""
		logging.info( "update_controls called" )
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
		self.last_click_text = None 

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

		#Create the output box
		self.tbox = QtGui.QTextEdit('MSIS output')
		vboxtopl.addWidget(self.tbox)
		self.update_controls() #Init the textbox

		#Create the plot button
		pbutton = QtGui.QPushButton('&Plot')
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
		self.comboXlab = QtGui.QLabel('Main X Var')
		self.comboXlabaux = QtGui.QLabel('Aux X Var')

		comboXVbox = QtGui.QVBoxLayout()
		comboXVbox.addWidget(self.comboXlab)
		comboXVbox.addWidget(self.comboXvar)
		comboXVbox.addWidget(self.comboXlabaux)
		comboXVbox.addWidget(self.comboXvaraux)
		
		self.comboYvar = QtGui.QComboBox()
		self.comboYvaraux = QtGui.QComboBox()
		self.comboYlab = QtGui.QLabel('Main Y Var')
		self.comboYlabaux = QtGui.QLabel('Aux Y Var')
		
		comboYVbox = QtGui.QVBoxLayout()
		comboYVbox.addWidget(self.comboYlab)
		comboYVbox.addWidget(self.comboYvar)
		comboYVbox.addWidget(self.comboYlabaux)
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
		self.auxcanv.refresh(force_autoscale=True)

	def canvas_key_pressed(self,event):
		"""Event handler for keyboard events while canvas has focus"""
		pass

	def canvas_clicked(self,event):
		"""Event handler for an matplotlib event, instead of a QT event"""

		#Ignore everything except left-clicks
		#right-clicks envoke a QT Menu
		if event.button!=1:
			return


		ax = event.inaxes #Which axes the click occured in
		if self.last_click is not None:
			for click in self.last_click:
				for ln in click:
					ln.set_visible(False)
		else:
			self.last_click = []

		if self.last_click_text is not None:
			self.last_click_text.set_visible(False)

		if ax is not None: #If the click occured in a axes
			canv = ax.figure.canvas

			#There's nothing easy a click can do on a multiple line axes,
			#so just do nothing
			if canv.pdh.xmulti or canv.pdh.ymulti:
				#give up and go home
				return

			#Click location
			x = event.xdata 
			y = event.ydata 
			if canv.controlstate['plottype']=='map':
				mx,my = x,y
				x,y = canv.pdh.map(mx,my,inverse=True)
			
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
			clz = zdata[ind]#Closest datapoint z coordinate
			
			clxtxt = '%.2e' % (clx) if np.abs(clx) < .01 or np.abs(clx) > 10000 else '%.2f' % (clx)
			clytxt = '%.2e' % (cly) if np.abs(cly) < .01 or np.abs(cly) > 10000 else '%.2f' % (cly)
			clztxt = '%.2e' % (clz) if np.abs(clz) < .01 or np.abs(clz) > 10000 else '%.2f' % (clz)
				
			if canv.controlstate['plottype']=='map': 
				#Plot point at clicked location and add resulting lines.line2D to the 'clicked' list
				#Add the plotted line to the clicked list
				self.last_click.append(canv.pdh.map.plot(clx,cly,'rx',markersize=5,latlon=True))
				self.last_click_text = ax.text(mx,my,clztxt,fontsize=4,color='red',fontweight='bold',\
					bbox={'facecolor':'white','edgecolor':'none','alpha':0.7, 'pad':1},ha='left',va='bottom')
				
				self.auxcanv.controlstate['lat']=clx
				self.auxcanv.controlstate['lon']=cly
				self.auxcanv.controlstate['yvar']='Altitude'
				self.auxcanv.controlstate['xvar']=self.maincanv.controlstate['zvar']
				self.auxcanv.controlstate['alt']=self.maincanv.controlstate['alt']
				self.set_locations(self.auxcanv.controlstate)
				self.set_aux_comboboxes(self.auxcanv.controlstate)
				self.auxcanv.refresh(force_autoscale=True)
				self.last_click[-1].append(self.auxcanv.ax.axhline(self.maincanv.controlstate['alt'],linestyle='-',linewidth=.5,color='b'))

			elif canv.controlstate['plottype']=='pcolor':
				#Plot point at clicked location and add resulting lines.line2D to the 'clicked' list
				#Add the plotted line to the clicked list
				self.last_click.append(ax.plot(clx,cly,'rx',markersize=5))
				self.last_click_text = ax.text(x,y,clztxt,fontsize=4,fontweight='bold',color='red',\
					bbox={'facecolor':'white','edgecolor':'none','alpha':0.7, 'pad':1},ha='left',va='bottom')
				#Decide which position axes is not represented in pcolor and plot that axes versus 
				#color variable on the auxillary canvas
				for var in ['Latitude','Longitude','Altitude']:
					if self.maincanv.controlstate['xvar']!=var and self.maincanv.controlstate['yvar']!=var:
						if var == 'Altitude':
							self.auxcanv.controlstate['yvar']=var
							self.auxcanv.controlstate['xvar']=self.maincanv.controlstate['zvar']
							self.auxcanv.controlstate['lat']=clx
							self.auxcanv.controlstate['lon']=cly
						elif var == 'Latitude':
							self.auxcanv.controlstate['xvar']='Latitude'
							self.auxcanv.controlstate['lat']=self.maincanv.controlstate['lat']
							self.auxcanv.controlstate['yvar']=self.maincanv.controlstate['zvar']
							if self.maincanv.controlstate['xvar']=='Longitude':
								self.auxcanv.controlstate['lon']=clx
								self.auxcanv.controlstate['alt']=cly
							else: #xvar must be altitude
								self.auxcanv.controlstate['lon']=cly
								self.auxcanv.controlstate['alt']=clx
						elif var == 'Longitude':
							self.auxcanv.controlstate['xvar']='Longitude'
							self.auxcanv.controlstate['lon']=self.maincanv.controlstate['lon']
							self.auxcanv.controlstate['yvar']=self.maincanv.controlstate['zvar']
							if self.maincanv.controlstate['xvar']=='Latitude':
								self.auxcanv.controlstate['lat']=clx
								self.auxcanv.controlstate['alt']=cly
							else: #xvar must be altitude
								self.auxcanv.controlstate['lat']=cly
								self.auxcanv.controlstate['alt']=clx
				self.set_locations(self.auxcanv.controlstate)
				self.set_aux_comboboxes(self.auxcanv.controlstate)
				self.auxcanv.refresh(force_autoscale=True)

				if self.auxcanv.controlstate['yvar']=='Altitude':
					self.last_click[-1].append(self.auxcanv.ax.axhline(self.maincanv.controlstate['alt'],linestyle='-',linewidth=.5,color='b'))
				elif self.auxcanv.controlstate['xvar']=='Latitude':
					self.last_click[-1].append(self.auxcanv.ax.axvline(self.maincanv.controlstate['lat'],linestyle='-',linewidth=.5,color='b'))
				elif self.auxcanv.controlstate['xvar']=='Longitude':
					self.last_click[-1].append(self.auxcanv.ax.axvline(self.maincanv.controlstate['lon'],linestyle='-',linewidth=.5,color='b'))

					
			else:
				self.last_click.append(ax.plot(clx,cly,'rx',markersize=5))
				self.last_click_text = ax.text(clx,cly,clztxt,fontweight='bold',color='red',\
					bbox={'facecolor':'white','edgecolor':'none','alpha':0.7, 'pad':1},ha='left',va='bottom')
				
			if canv.pdh.plottypes[canv.pdh.plottype]['gridxy']:
				clz = zdata[ind]#Closest datapoint z coordinate 
				#Show a message for each click in the app status bar
				self.statusBar().showMessage("%s:%s, %s:%s, %s:%s" % (canv.controlstate['xvar'],clxtxt,
																	  canv.controlstate['yvar'],clytxt,
																	  canv.controlstate['zvar'],clztxt), 4000)

			else:
				self.statusBar().showMessage("%s:%s, %s:%s" % (canv.controlstate['xvar'],clxtxt,
															   canv.controlstate['yvar'],clytxt), 4000)
				clz = np.nan
				
			#Redraw
			self.maincanv.draw()
			self.auxcanv.draw()
			

	def set_locations(self,controlstate):
		"""Sets all location controls to match the controlstate"""
		self.qlelat.setText('%.2f' % (controlstate['lat']))
		self.qlelon.setText('%.2f' % (controlstate['lon']))
		self.qlealt.setText('%.2f' % (controlstate['alt']))

	def set_comboboxes(self,controlstate):
		"""Sets the X, Y, and Z Variable Choosers By reading a canvas controlstate dict"""
		indX = self.comboXvar.findText(controlstate['xvar'])
		indY = self.comboYvar.findText(controlstate['yvar'])
		indZ = self.comboZvar.findText(controlstate['zvar'])
		self.comboXvar.setCurrentIndex(indX)
		self.comboYvar.setCurrentIndex(indY)
		self.comboZvar.setCurrentIndex(indZ)

	def set_aux_comboboxes(self,controlstate):
		"""Set the X and Y choosers (QComboBox) for the line plot by reading the controlstate"""
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

