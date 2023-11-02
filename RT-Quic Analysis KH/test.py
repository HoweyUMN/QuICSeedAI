import os
os.environ['R_HOME'] = 'C:\PROGRA~1\R\R-43~1.1'
import rpy2
import rpy2.robjects as robjects
from rpy2.ipython.ggplot import image_png
from rpy2.robjects.packages import importr, data

robjects.r('''
        library(stringr)
        library(readxl)
        library(tidyverse)
        library(ggplot2)
        library(QuICAnalysis) #QuICAnalysis 1.0

        # list all sub-folders in the current directory
        folders <- list.dirs(path = "./RT-Quic Analysis KH/data", recursive = FALSE)

        print(folders[1])

           ''')