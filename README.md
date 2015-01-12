# AtModExplorer
### _An (emperical) atmospheric model explorer_
This is a project of the University of Colorado Space Environment Data Analysis Group (SEDA), a part of the Colorado Center for Astrodynamics Research (CCAR).
_Currently in early development. There is a dependancy (the model code) which is not yet available publicly, so stay tuned_

### About 

The AtModExplorer is a PyQT4 based GUI for plotting real-time results from the Naval Research Laboratory Mass-Spec and Incoherent Scatter Radar Model of the neutral atmosphere ([NRLMSISE00](http://www.nrl.navy.mil/research/nrl-review/2003/atmospheric-science/picone/)). This model is maintained by NRL, and is used operationally and for research in the atmospheric and space sciences. 

This tool currently allows users to plot variables such as the mass density, temperature and number densities of various major atmospheric chemical constituants (O, O2, N, Ar, N2, etc.). It allows uses to plot these against various position coordinates such as latitude, longitude and altitude, on pseudocolor (heatmap) plots, line graphs, or on top of various map projections. 

### Installation
_Don't, yet. A few more major things need to be addressed. If you're interested in the project, please email me._

On linux systems, (tested so far on Ubuntu 14.04, running the Anaconda python distribution), first ensure you have the following dependancies:
* Gfortran (sudo apt-get install gfortran)
* Numpy
* Matplotlib 
* Basemap
* MsisPy - python (f2py wrapper) implementation of NRLMSISE00 __Not yet available. Will be added to PyPI soon__

Then:
```{sh}
git clone https://github.com/lkilcommons/atmodexplorer.git
cd atmodexplorer
python setup.py build
python setup.py install
```

### Running the GUI
```{python}
import atmodexplorer
atmodexplorer.__init__()
```
