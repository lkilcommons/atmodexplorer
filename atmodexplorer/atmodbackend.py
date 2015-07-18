import sys
import os
import random
import matplotlib.widgets as widgets
import matplotlib.axes

#Main imports
import numpy as np

#import pandas as pd
import sys, pdb, textwrap
import datetime
#sys.path.append('/home/liamk/mirror/Projects/geospacepy')
#import special_datetime, lmk_utils, satplottools #Datetime conversion class

from matplotlib.figure import Figure
import matplotlib as mpl
import textwrap #for __str__ of ModelRun

from mpl_toolkits.basemap import Basemap
from matplotlib import ticker
from matplotlib.colors import Normalize, LogNorm


import msispy,hwmpy
from collections import OrderedDict

import logging
logging.basicConfig(level=logging.INFO)

class ModelRunOD(OrderedDict):
	"""
	OrderedDict subclass for storing values for variables and drivers. It also stores an additional dict of units for each driver,
	and allowed values for each driver
	"""
	def __init__(self):
		super(ModelRunOD,self).__init__()
		self.log = logging.getLogger(self.__class__.__name__)
		self.descriptions = dict()
		self.units = dict()
		self.allowed_range = dict()

	def range_correct(self,key,val):
		"""Check that the value about to be set at key, is within the specified allowed_range for that key"""
		if isinstance(val,np.ndarray):
			outrange = np.logical_or(val < float(self.allowed_range[key][0]),val > float(self.allowed_range[key][1]))
			val[outrange] = np.nan
			if np.flatnonzero(outrange).shape[0] > 1:
				self.log.debug("%d values were out of range for key %s allowed_range=(%.3f-%.3f)" %(np.flatnonzero(outrange).shape[0],
					key,self.allowed_range[key][0],self.allowed_range[key][1]))
			else:
				self.log.debug("No values were out of range for key %s" % (key))
		elif isinstance(val,list):
			for k in range(len(val)):
				v = val[k]
				if v > self.allowed_range[key][1]:
					self.log.warn("Attempting to set %dth element of key %s to a value greater than allowed [%s]. setting to max allowed [%s]" % (k,
						key,str(v),str(self.allowed_range[key][1])))
					val[k] = self.allowed_range[key][1]
				elif v < self.allowed_range[key][0]:
					self.log.warn("Attempting to set %dth element of key %s to a value greater than allowed [%s] set to min allowed [%s]" % (k,
						key,str(v),str(self.allowed_range[key][0])))
					val[k] = self.allowed_range[key][0]
				elif v >= self.allowed_range[key][0] and v <= self.allowed_range[key][1]:
					pass
				else:
					raise RuntimeError("Nonsensical value in range_correct for index %d of %s: %s, allowed range is %s" % (int(k),key,str(val),str(self.allowed_range[key])))
		else: #assume it's a scalar value
			if val > self.allowed_range[key][1]:
				self.log.warn("Attempting to set key %s to a value greater than allowed [%s],setting to max allowed [%s]" % (key,str(val),str(self.allowed_range[key][1])))
				val = self.allowed_range[key][1]
			elif val < self.allowed_range[key][0]:
				self.log.warn("Attempting to set key %s to a value greater than allowed [%s],setting to min allowed [%s]" % (key,str(val),str(self.allowed_range[key][0])))
				val = self.allowed_range[key][0]
			elif val >= self.allowed_range[key][0] and val <= self.allowed_range[key][1]:
				pass
			else:
				raise RuntimeError("Nonsensical value in range_correct for %s: %s, allowed values: %s" % (key,str(val),str(self.allowed_range[key])))
		return val 

	def __setitem__(self,key,val):
		"""Check that we obey the allowed_range"""
		if key not in self.units:
			self.log.warn("ON SETTING %s has no units. Setting to None" %(key))
			self.units[key]=None
		if key not in self.descriptions:
			self.log.warn("ON SETTING %s has no description. Setting to None" %(key))
			self.descriptions[key]=None
		if key not in self.allowed_range:
			self.log.warn("ON SETTING %s has no allowed_range. Skipping range check" %(key))
		else:
			val = self.range_correct(key,val)

		OrderedDict.__setitem__(self,key,val)

	def __getitem__(self,key):
		item = OrderedDict.__getitem__(self,key)
		if key not in self.allowed_range:
			self.log.warn("ON GETTING %s has no allowed_range." %(key))
		if key not in self.descriptions:
			self.log.warn("ON GETTING %s has no description." %(key))
		if key not in self.units:
			self.log.warn("ON GETTING %s has no units." %(key))
		return item

	def copyasdict(self):
		newdict = dict()
		for key in self:
			newdict[key]=OrderedDict.__getitem__(self,key)
		return newdict
	
	def __call__(self):
		pass

	
class ModelRunDriversOD(ModelRunOD):
	def __init__(self):
		super(ModelRunDriversOD,self).__init__()
		self.awesome = True
		

class ModelRunVariablesOD(ModelRunOD):
	def __init__(self):
		super(ModelRunVariablesOD,self).__init__()
		#Add functionality for variable limits
		self.lims = dict()
		self._lims = dict()
		self.npts = dict()

class ModelRun(object):
	"""
	The ModelRun class is a generic class for individual calls to atmospheric models.

	The idea is to have individual model classes subclass this one, and add their specific
	run code to the 'populate method'.

	**The assumptions are:**
	
	* All atmospheric models take as input latitude, longitude and altitude
	* User will want data on a 2-d rectangular grid or as column arrays

	**The parameters used:**
	
	* **xkey** - string or None
		The key into vars,lims, and npts for the variable that represents the 1st dimension of the desired output (x-axis)
	* **ykey** - string or None
		The key into vars,lims, and npts for the variable that represents the 2nd dimension of the desired output (y-axis)

	* **vars** - an OrderedDict (Ordered because it's possible user will always want to iterate in a predetermined order over it's keys)
		#. The keys of vars are the names of the data stored in it's values. 
		#. vars always starts with keys 'Latitude','Longitude', and 'Altitude'

	* **lims** - an OrderedDict
		The range [smallest,largest] of a particular variable that will be used determine:
		#. The range of values of the independant variables (i.e. Latitude, Longitude or Altitude) the model will generate results format
		#. The range of values the user could expect a particular output variable to have (i.e to set axes bounds)

	* **npts** - an OrderedDict
		#. The number of distinct values between the associated lims of particular input variable that will be passed to the model
		i.e. (how the grid of input locations will be shaped and how big it will be)

	* **drivers** - a Dictionary
		Additional inputs that will be passed to the model (using **self.drivers in the model call).
		Inititialized to empty, set via the subclass

	Subclass best practices:
		* In the subclass __init__ method, after calling the superclass **__init__** method, the user should:
			* Set any keys in the self.drivers dict that will then be passed a keyword arguments to the model wrapper
		* In the populate method, after calling the superclass **populate** method, the user should:
			* Call the model using the flattened latitude, longitude, and altitude arrays prepared in the superclass method,
			and pass the drivers dict as keyword arguments.
		
	**Example for horizontal wind model subclass:**

	.. code-block:: python
		
		def __init__(self):
			#This syntax allows for multiple inheritance,
			#we don't use it, but it's good practice to use this 
			#instead of ModelRun.__init__()
			super(HWMRun,self).__init__()
			#ap - float
			#	daily AP magnetic index
			self.drivers['dt']=datetime.datetime(2000,6,21,12,0,0)
			self.drivers['ap']=None

		def populate():
			super(HWMRun,self).populate()
			
			self.winds,self.drivers = hwmpy.hwm(self.flatlat,self.flatlon,self.flatalt,**self.drivers)
			
			#Now add all the zonal and meridional winds to the dictionary
			for w in self.winds:
				self.vars[w] = self.winds[w]

			#Now make everything into the appropriate shape, if were
			#expecting grids. Otherwise make everything into a column vector
			if self.shape is None:
				self.shape = (self.npts,1)
				
			for v in self.vars:
				self.vars[v] = np.reshape(self.vars[v],self.shape)
				if v not in ['Latitude','Longitude','Altitude']:
					self.lims[v] = [np.nanmin(self.vars[v].flatten()),np.nanmax(self.vars[v].flatten())]

	**Operation works like this:**
		* Assume that we have a model run subclass call MyModel
		* Assume we have an instance of MyModel called mm 
		
		#. User (or calling method) decides that they want:
			* To plot a GLOBAL grid at an altitude of 110km that is Latitude (50 pts) vs. Longitude (75 pts) vs. Model output
		#. They set mm.npts['Latitude']=50 and mm.npts['Longitude']=75 to tell the object what the size of the grid is
		#. They call mm.set_x('Latitude'), and mm.set_y('Longitude') to set which dimensions correspond to which variables
		#. Since the model also requires an altitude value, they must set mm.vars['Altitude']=110 
		#. Since they want the grid to be global they set mm.lims['Latitude']=[-90.,90.] and mm.lims['Longitude']=[-180.,180.]
		#. Then they call mm.populate() to call the model for their desired grid

	**Calling:**
		* Getting a value from the ModelRun instance as if it were a dictionary i.e. mm['Latitude'], returns data,limits for 
			the variable 'Latitude'. Handles differencing any non position variables with another ModelRun instance at mm.peer if mm.peer is not None

	**Peering:**
		* the peer parameter can be set to another ModelRun instance to return difference between variables in two runs
		TODO: Document peering 

	"""
	def __init__(self):
		
		
		#Attributes which control how we do gridding
		#if only one is > 1, then we do vectors
		#
		self.modelname = None
		self.log = logging.getLogger(self.__class__.__name__)

		#Determines grid shape
		self.xkey = None
		self.ykey = None

		#the cannocical 'vars' dictionary, which has 
		#keys which are used to populate the combobox widgets,
		self.vars = ModelRunVariablesOD()
		self.vars['Latitude']=None
		self.vars['Longitude']=None
		self.vars['Altitude']=None

		#if two are > 1, then we do a grid
		self.vars.npts['Latitude']=1
		self.vars.npts['Longitude']=1
		self.vars.npts['Altitude']=1
		
		self.vars.lims['Latitude']=[-90.,90.]
		self.vars.lims['Longitude']=[-180.,180.]
		self.vars.lims['Altitude']=[0.,400.]

		#the _lims dictionary is very similar to the lims, but it is NOT TO BE MODIFIED
		#by any outside objects. It records the max and min values of every variable and is 
		#set when the finalize method is called. The reason it is included at all is so that
		#if the user (or another method) changes the lims, we can revert back to something if they
		#want that.
		for k in self.vars.lims:
			self.vars._lims[k] = self.vars.lims[k]
			self.vars.allowed_range[k] = self.vars.lims[k]

		#The units dictionary simply holds the unit for a variable
		self.vars.units['Latitude'] = 'deg'
		self.vars.units['Longitude'] = 'deg'
		self.vars.units['Altitude'] = 'km'

		#The drivers dictionary take input about whatever solar wind parameters drive the model
		#These must be either scalars (floats) or lists.
		#The keys to this dict must be keyword argument names in the model call 
		self.drivers = ModelRunDriversOD()
		self.log.debug("Class of drivers dict is %s" % (self.drivers.__class__.__name__))

		self.shape = None #Tells how to produce gridded output, defaults (if None) to use column vectors
		self.totalpts = None #Tells how many total points
		self.peer = None #Can be either None, or another ModelRun, allows for comparing two runs

	def autoscale_all_lims(self):
		for key in self.vars.lims:
			self.autoscale_lims(key)
			
	def autoscale_lims(self,key):
		if key in self.vars.lims and key in self.vars._lims:
			self.log.info("Restoring original bounds (was %s, now %s) for %s" % (str(self.vars.lims[key]),str(self.vars._lims[key]),key))
			self.vars.lims[key] = self.vars._lims[key]
		else:
			raise ValueError("Key %s is not a valid model run variable" % (key))

	def hold_constant(self,key):
		"""Holds an ephem variable constant by ensuring it's npts is 1s"""
		self.log.info("Holding %s constant" % (key))
		if key in ['Latitude','Longitude','Altitude']:
			self.vars.npts[key] = 1
		else:
			raise RuntimeError('Cannot hold %s constant, not a variable!'%(key))

	def set_x(self,key):
		"""Sets an emphem variable as x"""
		self.log.info("X is now %s" % (key)) 
		if key in ['Latitude','Longitude','Altitude']:
			self.xkey = key
		else:
			raise RuntimeError('Cannot set %s as x, not a variable!'%(key))

	def set_y(self,key):
		"""Sets an emphem variable as y"""
		self.log.info("Y is now %s" % (key))
		if key in ['Latitude','Longitude','Altitude']:
			self.ykey = key
		else:
			raise RuntimeError('Cannot set %s as y, not a variable!'%(key))

	def as_csv(self):
		pass

	def populate(self):
		"""Populates itself with data"""
		#Make sure that everything has the same shape (i.e. either a grid or a vector of length self.npts)
		self.log.info("Now populating model run") 
		
		#Count up the number of independant variables
		nindependent=0
		for key in self.vars.npts:
			if self.vars.npts[key] > 1:
				nindependent+=1
				self.shape = (self.vars.npts[key],1)

		if nindependent>1: #If gridding set the shape
			self.shape = (int(self.vars.npts[self.xkey]),int(self.vars.npts[self.ykey]))

		#Populate the ephemeris variables as vectors
		for var in ['Latitude','Longitude','Altitude']:
			if self.vars.npts[var]>1:
					self.vars[var] = np.linspace(self.vars.lims[var][0],self.vars.lims[var][1],self.vars.npts[var])
					self.log.debug("Generating %d %s points from %.3f to %.3f" % (self.vars.npts[var],var,self.vars.lims[var][0],self.vars.lims[var][1]))
			else:
				if self.vars[var] is not None:
					self.vars[var] = np.ones(self.shape)*self.vars[var]
				else:
					raise RuntimeError('Set %s to something first if you want to hold it constant.' % (var))
			
		if nindependent>1:
			x = self.vars[self.xkey]
			y = self.vars[self.ykey]
			X,Y = np.meshgrid(x,y)
			self.vars[self.xkey] = X
			self.vars[self.ykey] = Y

		#Now flatten everything to make the model call
		self.flatlat=self.vars['Latitude'].flatten()
		self.flatlon=self.vars['Longitude'].flatten()
		self.flatalt=self.vars['Altitude'].flatten()
		self.totalpts = len(self.flatlat)

	def finalize(self):
		"""
		Call after populate to finish shaping the data and filling the lims dict
		"""
		#Now make everything into the appropriate shape, if were
		#expecting grids. Otherwise make everything into a column vector
		if self.shape is None:
			self.shape = (self.npts,1)
			
		for v in self.vars:
			self.vars[v] = np.reshape(self.vars[v],self.shape)
			self.vars._lims[v] = [np.nanmin(self.vars[v].flatten()),np.nanmax(self.vars[v].flatten())]
			if v not in ['Latitude','Longitude','Altitude']: #Why do we do this again? Why not the positions?
				self.vars.lims[v] = [np.nanmin(self.vars[v].flatten()),np.nanmax(self.vars[v].flatten())]
			
			
	def __str__(self):
		"""
		Gives a description of the model settings used to make this run
		"""
		mystr = "Model: %s|" % (self.modelname)
		if self.xkey is not None:
			mystr = mystr+"Dimension 1 %s: [%.3f-%.3f][%s]|" % (self.xkey,self.vars.lims[self.xkey][0],self.vars.lims[self.xkey][1],self.vars.units[self.xkey])
		if self.ykey is not None:
			mystr = mystr+"Dimension 2 %s: [%.3f-%.3f][%s]|" % (self.ykey,self.vars.lims[self.ykey][0],self.vars.lims[self.ykey][1],self.vars.units[self.ykey])
		if 'Latitude' not in [self.xkey,self.ykey]:
			mystr = mystr+"Latitude held constant at %.3f|" % (self.vars['Latitude'].flatten()[0]) #By this point they will be arrays
		if 'Longitude' not in [self.xkey,self.ykey]:
			mystr = mystr+"Longitude held constant at %.3f|" % (self.vars['Longitude'].flatten()[0])
		if 'Altitude' not in [self.xkey,self.ykey]:
			mystr = mystr+"Altitude held constant at %.3f|" % (self.vars['Altitude'].flatten()[0])
		for d in self.drivers:
			mystr = mystr+"Driver %s: %s[%s]|" % (d,str(self.drivers[d]),str(self.drivers.units[d]))
		mystr = mystr+"Generated at: %s"  % (datetime.datetime.now().strftime('%c'))
		return mystr

	def __getitem__(self,key):
		"""Easy syntax for returning data"""
		if hasattr(key, '__iter__'): #If key is a sequence of some kind
			self.log.debug("Getting multiple variables/limits %s" % (str(key)))
			var = []
			lim = []
			unit = []
			desc = []
			for k in key:
				v,l,u,d = self.__getitem__(k)
				var.append(v)
				lim.append(l)
				unit.append(u)
				desc.append(d)
			return var,lim,unit,desc
		else:
			if self.peer is None:
				self.log.debug("Getting variables/limits/units/description for %s" % (key))
				return self.vars[key],self.vars.lims[key],self.vars.units[key],self.vars.descriptions[key]
			else:
				if key not in ['Latitude','Longitude','Altitude']:
					self.log.info( "Entering difference mode for var %s" % (key))
					#Doesn't make sense to difference locations
					mydata,mylims = self.vars[key],self.vars.lims[key]
					peerdata,peerlims = self.peer[key] #oh look, recursion opportunity!
					newdata = mydata-peerdata
					newlims = (np.nanmin(newdata.flatten()),np.nanmax(newdata.flatten()))
					newunits = 'diff(%s)' % str(self.vars.units[key])
					newdesc = self.vars.descriptions[key]+"(difference)"
					#newlims = lim_pad(newlims)
					return newdata,newlims,newunits,newdesc
				else:
					return self.vars[key],self.vars.lims[key]

class HWMRun(ModelRun):
	""" Class for individual calls to HWM """
	import hwmpy
	
	def __init__(self):
		"""Initialize HWM ModelRun Subclass"""
		super(HWMRun,self).__init__()
		#HWM DRIVERS
		#ap - float
		#	daily AP magnetic index
		#Overwrite the superclass logger
		self.log = logging.getLogger(__name__)
		
		self.modelname = "Horizontal Wind Model 07 (HWM07)"

		self.drivers['dt']=datetime.datetime(2000,6,21,12,0,0)
		self.drivers.allowed_range['dt'] = [datetime.datetime(1970,1,1),datetime.datetime(2012,12,31,23,59,59)]
		self.drivers.units['dt'] = 'UTC'
		
		self.drivers['ap']=None
		self.drivers.allowed_range['ap'] = [0,400]
		self.drivers.units['ap'] = 'unitless' #No units 

		self.vars.allowed_range['Altitude'] = [100.,500.]
		
	def populate(self):

		super(HWMRun,self).populate()
		
		self.log.info( "Now runing HWM07 for %s...\n" % (self.drivers['dt'].strftime('%c')))
		self.log.info( "Driver dict is %s\n" % (str(self.drivers)))

		#Call the F2Py Wrapper on the Fortran HWM07
		self.winds,outdrivers = hwmpy.hwm(self.flatlat,self.flatlon,self.flatalt,**self.drivers)
		
		#Copy the output drivers into the drivers dictionary
		for d in outdrivers:
			self.drivers[d] = outdrivers[d]

		#Now add all the zonal and meridional winds to the dictionary
		#Also add the units
		for w in self.winds:
			self.vars[w] = self.winds[w]
			self.vars.units[w] = 'km/s'

		#Finish reshaping the data
		self.finalize()

#MSIS DRIVERS
		#f107 - float
		#	daily f10.7 flux for previous day
		#ap_daily - float
		#	daily AP magnetic index
		#f107a - optional,float
		#	81 day average of f10.7 flux (centered on date)
		#ap3 - optional, float
		#	3 hour AP for current time
		#ap33 - optional, float
		#	3 hour AP for current time - 3 hours
		#ap36 - optional, float
		#	3 hour AP for current time - 6 hours
		#ap39 - optional, float
		#	3 hour AP for current time - 9 hours
		#apa1233 - optional, float
		#	Average of eight 3 hour AP indicies from 12 to 33 hrs prior to current time
		#apa3657 
		#	Average of eight 3 hour AP indices from 36 to 57 hours prior to current time

class MsisRun(ModelRun):
	""" Class for individual calls to NRLMSISE00 """
	import msispy
	
	def __init__(self):
		"""ModelRun subclass which adds MSIS to GUI"""
		super(MsisRun,self).__init__()
		
		#Overwrite the superclass logger
		self.log = logging.getLogger(self.__class__.__name__)

		self.modelname = "NRLMSISE00"
		self.drivers['dt']=datetime.datetime(2000,6,21,12,0,0)
		self.drivers.allowed_range['dt'] = [datetime.datetime(1970,1,1,0,0,0),datetime.datetime(2012,12,31,23,59,59)]
		self.drivers.descriptions['dt'] = 'Date and time of model run'

		self.drivers['f107']=None
		self.drivers.allowed_range['f107'] = [65.,350.]
		self.drivers.units['f107'] = 'SFU'
		self.drivers.descriptions['f107'] = 'Solar 10.7 cm Flux'

		self.drivers['ap_daily']=None
		self.drivers.allowed_range['ap_daily'] = [0.,400.]
		self.drivers.units['ap_daily'] = 'unitless' #No units 
		self.drivers.descriptions['ap_daily'] = 'AP planetary activity index'
		
		self.drivers['f107a']=None
		self.drivers.allowed_range['f107a'] = [65.,350.]
		self.drivers.units['f107a'] = 'SFU' #10^-22 W/m2/Hz'
		self.drivers.descriptions['f107a'] = '81-day Average Solar 10.7 cm Flux'
		
		#Warning: if you don't define this you will be restricted to 
		#0 to 400 km, which is the default set in the above function
		self.vars.allowed_range['Altitude'] = [0.,1000.]
		
	def populate(self):

		super(MsisRun,self).populate()
		
		self.log.info( "Now runing NRLMSISE00 for %s...\n" % (self.drivers['dt'].strftime('%c')))
		self.log.info( "Driver dict is %s\n" % (str(self.drivers)))

		self.species,self.t_exo,self.t_alt,outdrivers = msispy.msis(self.flatlat,self.flatlon,self.flatalt,**self.drivers)
		
		#Copy the output drivers into the drivers dictionary
		for d in outdrivers:
			self.drivers[d] = outdrivers[d]

		#Now add temperature the variables dictionary
		self.vars['T_exospheric'] = self.t_exo
		self.vars.units['T_exospheric'] = 'K'
		self.vars.descriptions['T_exospheric'] = 'Exospheric Temperature'
		
		self.vars['Temperature'] = self.t_alt
		self.vars.units['Temperature'] = 'K'
		
		#Now add all of the different number and mass densities to to the vars dictionary
		for s in self.species:
			self.vars[s] = self.species[s]
			if s == 'mass':
				self.vars.units[s] = 'g/cm^3'
				self.vars.descriptions[s] = 'Mass Density'
			else:
				self.vars.units[s] = '1/cm^3'
				self.vars.descriptions[s] = 'Number Density of %s' % (s)

		self.finalize()

	
class ModelRunner(object):
	""" Makes model calls """
	def __init__(self,canvas=None,model="msis"):
		self.log = logging.getLogger(self.__class__.__name__)

		self.cv = canvas
		self.model = model

		#Init the runs list (holds each run in sequence)
		self.runs = []
		#Start with a blank msis run
		self.init_nextrun()
		self.nextrun.drivers['dt'] = datetime.datetime(2000,6,21,12,0,0) #Summer solstice


		#Set counters
		self.n_total_runs=0
		self.n_max_runs=10
		self._lastind=-1 #The index of the current 'previous' model

		self._differencemode = False # Init the property

		#Create the dictionary that stores the settings for the next run
	@property
	def lastind(self):
		return self._lastind

	@lastind.setter
	def lastind(self, value):
		if self.lastind < 0 and self.lastind >= -1*len(self.runs):
			self._lastind = value
		else:
			self.log.warn("Attempted to set model history index to %s, which is invalid." % ( str(value)))
	
	def init_nextrun(self):
		if self.model.lower() == 'msis':
			self.nextrun = MsisRun()
		elif self.model.lower() == 'hwm':
			self.nextrun = HWMRun()
		else:
			raise ValueError("%s is not a valid model to run" % (self.model))

	def __call__(self,propagate_drivers=False):
		#Add to runs list, create a new nextrun
		self.runs.append(self.nextrun)
		self.init_nextrun()

		if propagate_drivers:
			for key in self.runs[-1].drivers:
				if key in self.nextrun.drivers and self.nextrun.drivers[key] is None:
					self.nextrun.drivers[key] = self.runs[-1].drivers[key]

		if self.differencemode:
			#Update the peering
			self.nextrun.peer = self.runs[-1].peer
		else:
			for run in self.runs:
				run.peer = None

		self.n_total_runs+=1
		self.log.info( "Model run number %d added." % (self.n_total_runs))
		if len(self.runs)>self.n_max_runs:
			del self.runs[0]
			self.log.info( "Exceeded total number of stored runs %d. Run %d removed" %(self.n_max_runs,self.n_total_runs-self.n_max_runs))

	#Implement a simple interface for retrieving data, which is opaque to whether or now we're differencing
	#or which model we're using
	def __getitem__(self,key):
		"""Shorthand for self.runs[-1][key], which returns self.runs[-1].vars[key],self.runs[-1].lims[key]"""
		return self.runs[self.lastind][key]

	def __setitem__(self,key,value):
		"""Shorthand for self.nextrun[key]=value"""
		self.nextrun.key=value

	#Make a property for the difference mode turning on or off, which automatically updates the last run peer
	@property
	def differencemode(self):
		return self._differencemode

	@differencemode.setter
	def differencemode(self, boo):
		self.log.info( "Difference mode is now %s" % (str(boo)))
		if boo:
			self.nextrun.peer = self.runs[-1]
		else:
			self.nextrun.peer = None

		self._differencemode = boo
		

class PlotDataHandler(object):
	def __init__(self,canvas,controlstate=None,plottype='line',cscale='linear',mapproj='moll'):
		"""
		Takes a singleMplCanvas instance to associate with
		and plot on
		"""
		self.canvas = canvas
		self.log = logging.getLogger(self.__class__.__name__)
		if controlstate is None:
			self.controlstate = canvas.controlstate #This is for atmodexplorer, where there's only one controlstate 
			#and it's associated with the canvas
		else:
			self.controlstate = controlstate # This is for atmodweb, where the controlstate is associated with the synchronizer

		self.fig = canvas.fig
		self.ax = canvas.ax
		self.axpos = canvas.ax.get_position()
		pts = self.axpos.get_points().flatten() # Get the extent of the bounding box for the canvas axes
		self.cbpos = None
		self.cb = None
		self.map_lw=.5 #Map linewidth
		self.map = None #Place holder for map instance if we're plotting a map
		#The idea is to have all plot settings described in this class,
		#so that if we want to add a new plot type, we only need to modify 
		#this class
		self.supported_projections={'mill':'Miller Cylindrical','moll':'Mollweide','ortho':'Orthographic'}
		self.plottypes = dict()
		self.plottypes['line'] = {'gridxy':False,'overplot_ready':True,'x_allowed':['all'],'y_allowed':['all'],'z_allowed':['none']}
		self.plottypes['pcolor'] = {'gridxy':True,'overplot_ready':False,'x_allowed':['position'],'y_allowed':['position'],'z_allowed':['notposition']}
		self.plottypes['map'] = {'gridxy':True,'overplot_ready':False,'x_allowed':['Longitude'],'y_allowed':['Latitude'],'z_allowed':['notposition']}
		if plottype not in self.plottypes:
			raise ValueError('Invalid plottype %s! Choose from %s' % (plottype,str(self.plottypes.keys())))
		
		#Assign settings from input
		self.plottype = plottype
		self.mapproj = mapproj # Map projection for Basemap
			
		#Init the data variables
		self.clear_data()

	def caption(self):
		#Generates a description of the graph
		cap = "%s" % (str(self.xname) if not self.xlog else "log(%s)" % (str(self.xname)))
		cap = cap + " vs. %s" % (str(self.yname) if not self.ylog else "log(%s)" % (str(self.yname)))
		if self.plottype == 'pcolor':
			cap = "Pseudocolor plot of "+ cap + " vs. %s" % (str(self.zname) if not self.zlog else "log(%s)" % (str(self.zname)))
		elif self.plottype == 'map':
			cap = "%s projection map of " % (self.supported_projections[self.mapproj])+ cap + \
					" vs. %s" % (str(self.zname) if not self.zlog else "log(%s)" % (str(self.zname)))
		return cap
			
	def clear_data(self):
		self.log.info("Clearing PlotDataHandler data NOW.")
		self.x,self.y,self.z = None,None,None #must be np.array
		self.xname,self.yname,self.zname = None,None,None #must be string
		self.xbounds,self.ybounds,self.zbounds = None,None,None #must be 2 element tuple
		self.xlog, self.ylog, self.zlog = False, False, False
		self.xunits, self.yunits, self.zunits = None, None, None
		self.xdesc, self.ydesc, self.zdesc = None, None, None
		self.npts = None
		self.statistics = None # Information about each plotted data

	def associate_data(self,varxyz,vardata,varname,varbounds,varlog,multi=False,units=None,description=None):
		#Sanity check 
		if not multi:
			thislen = len(vardata.flatten()) 
			thisshape = vardata.shape
		else:
			thislen = len(vardata[0].flatten()) 
			thisshape = vardata[0].shape
			#Make sure all the arguments have same length, even if left as default
			if not hasattr(units,'__iter__') : #if it isn't a list, make it one
				units = [units for i in range(len(vardata))]
			if not hasattr(description,'__iter__') : #if it isn't a list, make it one
				description = [description for i in range(len(vardata))]
			
		#Check total number of points
		if self.npts is not None:
			if thislen != self.npts:
				raise RuntimeError('Variable %s passed for axes %s had wrong flat length, got %d, expected %d' % (varname,varaxes,
					thislen,self.npts))

		#Check shape
			for v in [self.x[0] if self.multix else self.x,self.y[0] if self.multiy else self.y,self.z]:

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
			self.xmulti = multi
			self.xunits = units
			self.xdesc = description
		elif varxyz in ['y','Y',1,'yvar']:
			self.y = vardata
			self.yname = varname
			self.ybounds = varbounds
			self.ylog = varlog
			self.ymulti = multi
			self.yunits = units
			self.ydesc = description
		elif varxyz in ['z','Z',2,'C','c','zvar']:
			self.z = vardata
			self.zname = varname
			self.zbounds = varbounds
			self.zlog = varlog
			self.zunits = units
			self.zdesc = description
		else:
			raise ValueError('%s is not a valid axes for plotting!' % (str(varaxes)))

		

	def compute_statistics(self):
		self.statistics = OrderedDict()
		if self.plottype=='line':
			if not self.xmulti:
				self.statistics['Mean-%s'%(self.xname)]=np.nanmean(self.x)
				self.statistics['Median-%s'%(self.xname)]=np.median(self.x)
				self.statistics['StDev-%s'%(self.xname)]=np.nanstd(self.x)
			else:
				for n in range(len(self.x)):
					self.statistics['Mean-%s'%(self.xname[n])]=np.nanmean(self.x[n])
					self.statistics['Median-%s'%(self.xname[n])]=np.median(self.x[n])
					self.statistics['StDev-%s'%(self.xname[n])]=np.nanstd(self.x[n])

			if not self.ymulti:	
				self.statistics['Mean-%s'%(self.yname)]=np.nanmean(self.y)
				self.statistics['Median-%s'%(self.yname)]=np.median(self.y)
				self.statistics['StDev-%s'%(self.yname)]=np.nanstd(self.y)
			else:
				for n in range(len(self.y)):
					self.statistics['Mean-%s'%(self.yname[n])]=np.nanmean(self.y[n])
					self.statistics['Median-%s'%(self.yname[n])]=np.median(self.y[n])
					self.statistics['StDev-%s'%(self.yname[n])]=np.nanstd(self.y[n])

		elif self.plottype=='map' or self.plottypes=='pcolor':
			self.statistics['Mean-%s'%(self.zname)]=np.nanmean(self.z)
			self.statistics['Median-%s'%(self.zname)]=np.median(self.z)
			self.statistics['StDev-%s'%(self.zname)]=np.nanstd(self.z)
			if self.plottype=='map':
				self.statistics['Geo-Integrated-%s'%(self.zname)]=self.integrate_z()
	
	def integrate_z(self):
		#If x and y are longitude and latitude
		#integrates z over the grid
		if self.xname=='Longitude':
			lon=self.x.flatten()
		elif self.yname=='Longitude':
			lon=self.y.flatten()
		else:
			return np.nan
		if self.xname=='Latitude':
			lat=self.x.flatten()
		elif self.yname=='Latitude':
			lat=self.y.flatten()
		else:
			return np.nan
		#a little hack to get the altitude
		alt = self.controlstate['alt']
		r_km = 6371.2+alt

		zee = self.z.flatten()
		zint = 0.
		for k in xrange(len(lat)-1):
			theta1 = (90.-lat[k])/180.*np.pi
			theta2 = (90.-lat[k+1])/180.*np.pi
			dphi = (lon[k+1]-lon[k])/180.*np.pi
			zint +=  (zee[k]+zee[k+1])/2.*np.abs(r_km**2*dphi*(np.cos(theta1)-np.cos(theta2)))#area element
		return zint

	def plot(self,*args,**kwargs):

		if self.map is not None:
			self.map = None #Make sure that we don't leave any maps lying around if we're not plotting maps
		
		#print "self.ax: %s\n" % (str(self.ax.get_position()))

		#print "All axes: \n" 
		#for i,a in enumerate(self.fig.axes):
		#	print "%d: %s" % (i,str(a.get_position()))
		
		if self.statistics is None:
			self.log.debug("Computing statistics")
			self.compute_statistics()

		if self.cb is not None:
			#logging.info("removing self.cb:%s\n" % (str(self.cb.ax.get_position()))
			self.log.debug("Removing self.cb")
			self.cb.remove()
			self.cb = None


		self.ax.cla()
		self.fig.suptitle('')

		#self.zbounds = (np.nanmin(self.z),np.nanmax(self.z))
		
		if self.zlog:
			self.log.debug("Z var set to use log scale")
			self.z[self.z<=0.] = np.nan
			norm = LogNorm(vmin=self.zbounds[0],vmax=self.zbounds[1]) 
			locator = ticker.LogLocator()
			formatter = ticker.LogFormatter(10, labelOnlyBase=False) 
		else:
			norm = Normalize(vmin=self.zbounds[0],vmax=self.zbounds[1])
	
		if self.plottype == 'line':
			self.log.debug("Plottype is line")
			
			#Plot a simple 2d line plot
			if self.cb is not None:
				self.cb.remove()
				self.cb = None
				#self.ax.set_position(self.axpos)

			if not self.xmulti and not self.ymulti: #No overplotting
				
				self.log.debug("No multiplotting is requested: X var %s, Y var %s" % (str(self.xname),str(self.yname)))
				self.ax.plot(self.x,self.y,*args,**kwargs)
				xbnds = self.xbounds
				ybnds = self.ybounds
				xnm = self.xname  
				xnm += '' if self.xunits is None else '[%s]' % (str(self.xunits))
				ynm = self.yname 
				ynm += '' if self.yunits is None else '[%s]' % (str(self.yunits))

			elif self.xmulti and not self.ymulti: #Overplotting xvars
				
				self.log.debug("X multiplotting is requested: X vars %s, Y var %s" % (str(self.xname),str(self.yname)))
				xbnds = self.xbounds[0]
				ybnds = self.ybounds
				xnm = ''
				endut = self.xunits[0]
				for i in range(len(self.xname)):
					nm,ut = self.xname[i],self.xunits[i]
					if ut is not None and ut != endut:
						xnm += nm+'[%s]'%(str(ut))+','
					else:
						xnm += nm+','
				xnm = xnm[:-1] #Remove last comma
				xnm += '[%s]' % (endut) if endut is not None else ''
				
				print xnm
				print self.xbounds

				ynm = self.yname 
				ynm += '' if self.yunits is None else '[%s]' % (str(self.yunits))

				for i in range(len(self.x)):
					self.ax.plot(self.x[i],self.y,label=self.xname[i],*args,**kwargs) #should cycle through colors
					self.ax.hold(True)
					
					#Compute new bounds as incuding all bounds
					xbnds[0] = xbnds[0] if xbnds[0]<self.xbounds[i][0] else self.xbounds[i][0]
					xbnds[1] = xbnds[1] if xbnds[1]>self.xbounds[i][1] else self.xbounds[i][1]

			elif self.ymulti and not self.xmulti: #Overplotting yvars

				self.log.debug("Y multiplotting is requested: X var %s, Y vars %s" % (str(self.xname),str(self.yname)))
				ybnds = self.ybounds[0]
				xbnds = self.xbounds
				xnm = self.xname 
				xnm += '' if self.xunits is None else '[%s]' % (str(self.xunits))

				ynm = ''

				endut = self.yunits[0]
				for i in range(len(self.yname)):
					nm,ut = self.yname[i],self.yunits[i]
					if ut is not None and ut != endut:
						ynm += nm+'[%s]'%(str(ut))+','
					else:
						ynm += nm+','
				ynm = ynm[:-1] #Remove last comma
				ynm += '[%s]' % (endut) if endut is not None else ''

				for i in range(len(self.y)):
					self.ax.plot(self.x,self.y[i],label=self.yname[i],*args,**kwargs) #should cycle through colors
					self.ax.hold(True)
					#Compute new bounds as incuding all bounds
					
					print self.ybounds[i]
					ybnds[0] = ybnds[0] if ybnds[0]<self.ybounds[i][0] else self.ybounds[i][0]
					ybnds[1] = ybnds[1] if ybnds[1]>self.ybounds[i][1] else self.ybounds[i][1]

			#Set axes appearance and labeling
			self.ax.set_xlabel(xnm)
			if self.xlog:
				self.ax.set_xscale('log',nonposx='clip')
				self.ax.get_xaxis().get_major_formatter().labelOnlyBase = False
				
			self.ax.set_xlim(xbnds)
			self.log.debug("Setting bounds for X var %s, %s" % (str(self.xname),str(xbnds)))
				
			self.ax.set_ylabel(ynm)
			if self.ylog:
				self.ax.set_yscale('log',nonposx='clip')
				self.ax.get_yaxis().get_major_formatter().labelOnlyBase = False
				
			self.ax.set_ylim(ybnds)
			self.log.debug("Setting bounds for Y var %s, %s" % (str(self.yname),str(ybnds)))

			if self.xmulti or self.ymulti:
				self.ax.legend()

				#self.ax.set_ylim(0,np.log(self.ybounds[1]))
			self.ax.set_title('%s vs. %s' % (xnm if not self.xlog else 'log(%s)'%(xnm),ynm if not self.ylog else 'log(%s)'%(ynm)))
			self.ax.grid(True,linewidth=.1)
			if not self.xlog and not self.ylog:
				self.ax.set_aspect(1./self.ax.get_data_ratio())

			self.ax.set_position([.15,.15,.75,.75])
			#try:
			#	self.fig.tight_layout()
			#except:
			#	print "Tight layout for line failed"
		
		elif self.plottype == 'pcolor':
			self.log.info("Plottype is pcolor for vars:\n--X=%s lims=(%s)\n--Y=%s lims=(%s)\n--C=%s lims=(%s)" % (str(self.xname),str(self.xbounds),
				str(self.yname),str(self.ybounds),str(self.zname),str(self.zbounds)))

			xnm = self.xname
			xnm += '' if self.xunits is None else '[%s]' % (str(self.xunits))
			ynm = self.yname 
			ynm += '' if self.yunits is None else '[%s]' % (str(self.yunits))
			znm = self.zname
			znm += '' if self.zunits is None else '[%s]' % (str(self.zunits))

			mappable = self.ax.pcolormesh(self.x,self.y,self.z,norm=norm,shading='gouraud',**kwargs)
			#m.draw()

			self.ax.set_xlabel(xnm)
			if self.xlog:
				self.ax.set_xscale('log',nonposx='clip')
			self.ax.set_xlim(self.xbounds)

			self.ax.set_ylabel(ynm)
			
			if self.ylog:
				self.ax.set_xscale('log',nonposx='clip')
			self.ax.set_ylim(self.ybounds)
			
			if self.zlog: #Locator goes to ticks argument
				self.cb = self.fig.colorbar(mappable,ax=self.ax,orientation='horizontal',format=formatter,ticks=locator)
			else:
				self.cb = self.fig.colorbar(mappable,ax=self.ax,orientation='horizontal')

			#self.ax.set_position(self.axpos)
			#self.cb.ax.set_position(self.cbpos)
			self.cb.ax.set_position([.1,0,.8,.15])
			self.ax.set_position([.1,.25,.8,.7])

			self.cb.set_label(znm)
			self.ax.set_aspect(1./self.ax.get_data_ratio())
			self.fig.suptitle('%s vs. %s (color:%s)' % (xnm,ynm,
				znm if not self.zlog else 'log(%s)'% znm))
			
		elif self.plottype == 'map':
			#Basemap is too screwy for partial maps
			#self.ybounds = [-90.,90.]
			#self.xbounds = [-180.,180.]
			
			znm = self.zname
			znm += '' if self.zunits is None else '[%s]' % (str(self.zunits))

			self.log.info("Plottype is %s projection MAP for vars:\n--X=%s lims=(%s)\n--Y=%s lims=(%s)\n--C=%s lims=(%s)" % (str(self.mapproj),str(self.xname),str(self.xbounds),
				str(self.yname),str(self.ybounds),str(self.zname),str(self.zbounds)))

			if self.mapproj=='moll':
				m = Basemap(projection=self.mapproj,llcrnrlat=int(self.ybounds[0]),urcrnrlat=int(self.ybounds[1]),\
					llcrnrlon=int(self.xbounds[0]),urcrnrlon=int(self.xbounds[1]),lat_ts=20,resolution='c',ax=self.ax,lon_0=0.)
			elif self.mapproj=='mill':
				m = Basemap(projection=self.mapproj,llcrnrlat=int(self.ybounds[0]),urcrnrlat=int(self.ybounds[1]),\
					llcrnrlon=int(self.xbounds[0]),urcrnrlon=int(self.xbounds[1]),lat_ts=20,resolution='c',ax=self.ax)
			elif self.mapproj=='ortho':
				m = Basemap(projection='ortho',ax=self.ax,lat_0=int(self.controlstate['lat']),
					lon_0=int(self.controlstate['lon']),resolution='l')
			
			m.drawcoastlines(linewidth=self.map_lw)

			#m.fillcontinents(color='coral',lake_color='aqua')
			
			# draw parallels and meridians.
			m.drawparallels(np.arange(-90.,91.,15.),linewidth=self.map_lw)
			m.drawmeridians(np.arange(-180.,181.,30.),linewidth=self.map_lw)
			if self.zlog:
				mappable = m.pcolormesh(self.x,self.y,self.z,linewidths=1.5,latlon=True,norm=norm,
					vmin=self.zbounds[0],vmax=self.zbounds[1],shading='gouraud',**kwargs)
			else:
				mappable = m.pcolormesh(self.x,self.y,self.z,linewidths=1.5,latlon=True,norm=norm,
					vmin=self.zbounds[0],vmax=self.zbounds[1],shading='gouraud',**kwargs)
				
			
			latbounds = [self.ybounds[0],self.ybounds[1]]
			lonbounds = [self.xbounds[0],self.xbounds[1]]

			lonbounds[0],latbounds[0] = m(lonbounds[0],latbounds[0])
			lonbounds[1],latbounds[1] = m(lonbounds[1],latbounds[1])

			#self.ax.set_ylim(latbounds)
			#self.ax.set_xlim(lonbounds)

			#m.draw()


			if self.zlog:
				self.cb = self.fig.colorbar(mappable,ax=self.ax,orientation='horizontal',format=formatter,ticks=locator)
			else:
				self.cb = self.fig.colorbar(mappable,ax=self.ax,orientation='horizontal')

			#m.set_axes_limits(ax=self.canvas.ax)
			
			if self.mapproj == 'mill':
				self.ax.set_xlim(lonbounds)
				self.ax.set_ylim(latbounds)
				self.cb.ax.set_position([.1,.05,.8,.15])
				self.ax.set_position([.1,.2,.8,.7])
			elif self.mapproj == 'moll':
				self.cb.ax.set_position([.1,.03,.8,.13])
				self.ax.set_position([.05,.2,.9,.9])

			self.cb.set_label(znm)
			self.ax.set_title('%s Projection Map of %s' % (self.supported_projections[self.mapproj],
				znm if not self.zlog else 'log(%s)' % (znm)))
			self.map = m
		#Now call the canvas cosmetic adjustment routine
		self.canvas.apply_lipstick()
