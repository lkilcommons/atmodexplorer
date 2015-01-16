"""
AtModExplorer
This tool is designed for the use of students of atmospheric and space science and other interested parties.
This tool is being actively developed. Don't expect it to be correct, or even work right now.
No warranties, no liability, as per the LICENSE.txt.
"""
from atmodexplorer import *
import sys

def __init__():
	qApp = QtGui.QApplication(sys.argv)
	aw = AtModExplorerApplicationWindow()
	aw.show()
	sys.exit(qApp.exec_())


