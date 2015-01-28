#################
Code Architecture
#################

This GUI is intended to be as simple and extensible as possible. Therefore the class structure is designed around the GUI elements, with some background classes to handle data management in a sensible way. 

Class Organization
==================

* AtModExplorerApplicationWindow_ is the GUI itself. It extends QtGui.QMainWindow. Pretty much everything is a member of this class. Most especially, this class has all of the QT controls as members. It's methods are organized around getting information into and out of the QT widgets, as well as handling any GUI-wide events (i.e. clicks on either the main or auxillary canvases), or cross-canvas events (like clicking on the main canvas map to get the altitude profile on the auxillary canvas).
* mainCanvas_ and auxCanvas_ are thin subclasses of singleMplCanvas_ which is the workhorse of the whole program. The singleMplCanvas is a subclass of matplotlib's backends.backend_qt4agg.FigureCanvasQTAgg, which is basically the Qt version of the canvas class which contains and manages a matplotlib figure. Each of the canvas classes has associated with it a plotDataHandler_ instance and ModelRunner_ instance, which handle plotting and getting data respectively. The most important member of these classes is the 'controlstate' dictionary. This dictionary hold all of the information required to run the model per user specification of date, or position (latitude,longitude,altitude), as well as all information as to what to plot (i.e. what variable goes on the x axis, are we plotting a map or a pseudocolor plot, etc.). The mainCanvas_ and auxCanvas_ classes only include one additional method beyond those of the superclass: apply_lipstick(), which customizes the look of that canvas for it's location on the GUI.
* plotDataHandler_ is in charge of keeping track of what data is currently plotted on the canvas, the bounds of it's axes, the label for each axes, and especially what type of plot is to be plotted. It has the methods which actually call 'plot'. 
* ModelRunner_ is in charge of coordinating and storing different model runs. It maintains a 'runs' list of ModelRun_ objects, each of which represent a single run of the model for new inputs. Examples of user requests which will trigger a model run include: changing from a 3-d plot type to a 2-d one, or changing the date. Simply changing the variable will not initiate a new model run, nor will, for example, changing any visual aspect of the figures. 

.. currentmodule:: atmodexplorer

QT Application Window
=====================

This is the main GUI window


.. _AtModExplorerApplicationWindow:

.. autoclass:: AtModExplorerApplicationWindow
    :members:


Canvas Classes
==============

This subclass (actually it's subclasses, mainCanvas and auxCanvas) represent the main 'panels' on which the GUI can plot 

.. _singleMplCanvas:

.. autoclass:: singleMplCanvas
    :members:

.. _mainCanvas:

.. autoclass:: mainCanvas
	:members

.. _auxCanvas:

.. autoclass:: auxCanvas
	:members

Plotting Handler Class
======================

.. _plotDataHandler:

.. autoclass:: plotDataHandler
    :members:

Model Output Handling Classes
=============================

.. _ModelRun:

.. autoclass:: ModelRun
    :members:


.. _MsisRun:

.. autoclass:: MsisRun
    :members:

.. _ModelRunner:

.. autoclass:: ModelRunner
    :members:
 

