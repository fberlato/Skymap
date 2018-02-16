import matplotlib
matplotlib.use('Agg')

from mpi4py import MPI
mpi=MPI.COMM_WORLD
rank = mpi.Get_rank()

from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from gbmgeometry import *
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from trigdat_reader import TrigReader, Palantir
from threeML import *
from glob import glob
import warnings
warnings.simplefilter('ignore')
import os


trigger='170817529'

res = load_analysis_results('loc_results_multinest_mpi.fits')
pp = Palantir(res,trigdat='glg_trigdat_all_bn'+trigger+'_v01.fit')
skymap_plot = pp.skymap('n0','n1','n2','n3','n4','n5','n7','nb',cmap='viridis',show_earth=True)

skymap_plot.savefig('skymap_plot.png')

