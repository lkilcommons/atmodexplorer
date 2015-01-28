#########
Workflows
#########

While the AtModExplorer interface is highly general and should be amenable to many possible applications,
various common tasks in analyzing model output have been expressly addressed in the design.
 
Viewing altitude profiles for various latitudes and longitudes
==============================================================
The default mode of operation for the AtModExplorer is to plot contours over a map projection in the main panel. Clicking on a location on the map will display the altitude profile of the variable being 'contoured'  in the secondary panel.

Looking at the change in a variable caused by the change in an input parameter
==============================================================================
The AtModExplorer has a 'memory' of recent model runs (each run is it's own python object) and the 'difference mode' plots the difference between the selected variable in the most recent and second most recent runs of the model. For example, say that one wished to see how the altitude profile of temperature varied between the equator and the pole. One way to address that question would be to first plot the temperature (as Z/the color variable) versus longitude(X) and altitude(Y) using the 'pseudocolor' setting for plot-type. The latitude text box would be set to 0 for the first run, and then to 85. Then, entering difference-mode by marking the checkbox would cause the location-by-location difference in temperature it be plotted as the color variable for the pseudocolor plot. This mode works for all plot-types, and is very useful for determining the effect of changing input parameters.

Plotting multiple variables on the same axes
============================================
AtModExplorer has support for overplotting for 2-d line plots, simply by right clicking on the plot in question and selecting the desired variable from the dropdown 'overplot' menu. Upon selecting the variable, the plot will automatically refresh and the new variable will be plotted over the original in a different color. The number of variables one can overplot is technically unlimited, but a sensible upper limit for legibility is six to eight.

Determining statistics of a variable
====================================
One of the menu items available when a user right-clicks on one of the plots is 'statistics'. Selecting this item will pop up a dialog which displays various statistics about the data being plotted. This is useful when one wishes to know, say, the global average of a variable at a particular altitude. This would be accomplished by first plotting a 'map' plot-type with color variable of 'temperature', setting the altitude text-box to whatever altitude is desired (say, 110 km, for the ionosphere), and then right-clicking on the resulting plot and selecting 'statistics'.