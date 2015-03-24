############################
Data Management and Plotting
############################

The classes which handle running models, storing the output and plotting the user's desired info are completely independent of the GUI.

The model class ModelRun_ is a general one-instance-per-run superclass, which is then subclassed for different models, such as MSISRun_ or HWMRun_. This architecture allows serveral interesting behaviors.
* Peering: One ModelRun_ subclasses can be *peered* to another (one way, if A is peered to B, B is not nessecarily peered to A), which by default differences any common variable that the user calls for using the ``__getitem__`` magic method. An example of when this is useful is when the user has run a model for two different days over a global grid, and wishes to see which localle experienced the greatest change in a particular variable.
* History: By keeping ModelRun_ instances in a list, the ModelRunner_ class can maintain a history of recent runs, and recall any one of them at need.   

.. currentmodule:: atmodbackend

Plotting Handler Class
======================

.. _PlotDataHandler:

.. autoclass:: PlotDataHandler
    :members:

Model Output Handling Classes
=============================

.. _ModelRun:

.. autoclass:: ModelRun
    :members:

.. _MsisRun:

.. autoclass:: MsisRun
    :members:

.. _HWMRun:

.. autoclass:: HWMRun
    :members:

.. _ModelRunner:

.. autoclass:: ModelRunner
    :members: