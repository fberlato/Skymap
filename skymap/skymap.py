import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import matplotlib.colors as clrs
from matplotlib.patches import CirclePolygon
from matplotlib.collections import PatchCollection

from os import getcwd

from astropy.coordinates import SkyCoord
import astropy.units as un
from astropy.coordinates import get_sun
from astropy.time import Time

from threeML.analysis_results import load_analysis_results
from trigdat_reader import TrigReader
import gbmgeometry as gbmgeo


all_det_list = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']


class Skymap(object):
	
	def __init__(self, analysis_path, trigdat_path, real_coord=None, det_list=all_det_list, save_path=None, sun=True):

		"""
		:param analysis_file: path of the file with the bayesian analysis result.
		:param trigdat_path: path of the trigdat file.
		:param real_coord: real or reference equatorial coordinates for the source in the format [ra,dec] (in deg). Default is none.
		:param det_list: list of the detector you wish to plot, in the format ['n1','n5','b0']. Default is all detectors.
		:param save_path: path and name for the plotted skymap (default is 'skymap.pdf' in cwd)
                :param sun: plot sun position in the skymap (default is True)
		""" 

		self._analysis_path = analysis_path
		self._trigdat_path = trigdat_path

		if real_coord!=None:
			self._real_coord = [np.radians(-real_coord[0] + 180.), np.radians(real_coord[1])]
		else:
			self._real_coord = None

		self._det_list = det_list

		if save_path == None:
			self._save_path = getcwd()+'/skymap.pdf'
                else:
                        self._save_path = save_path
                
		results = load_analysis_results(self._analysis_path)
		self._ra_samples = np.radians(-np.array(results.samples[0]) + 180.)
		self._dec_samples = np.radians(np.array(results.samples[1]))

		ra_fit = np.radians(-results.get_data_frame().value[0] + 180.)
		dec_fit = np.radians(results.get_data_frame().value[1])

		self._fit_coord = [ra_fit, dec_fit]

		ra_cl_50 = results.get_equal_tailed_interval('test.position.ra',cl=0.5)
		dec_cl_50 = results.get_equal_tailed_interval('test.position.dec',cl=0.5)
		ra_cl_90 = results.get_equal_tailed_interval('test.position.ra',cl=0.9)
		dec_cl_90 = results.get_equal_tailed_interval('test.position.dec',cl=0.9)

		self._cl_50 = [np.radians(ra_cl_50), np.radians(dec_cl_50)]
		self._cl_90 = [np.radians(ra_cl_90), np.radians(dec_cl_90)]

		self._detectors = self._get_gbm_geometry()
		self._earth_circle = self._get_earth_circle()

	def plot_skymap(self):

		fig = plt.figure(figsize=(12,8))
		ax = fig.add_subplot(111, projection="mollweide")

		# we define some custom colors to use
		transp_black = clrs.ColorConverter().to_rgba('black', alpha=0.6)
		transp_green = clrs.ColorConverter().to_rgba('green', alpha=0.4)
		transp_grey = clrs.ColorConverter().to_rgba('grey', alpha=0.5)

		# get probability associated with each sampled point
		prob = self._get_probability_array(pix=64)

		cm = plt.cm.get_cmap('viridis')
		sampling_points = ax.scatter(self._ra_samples, self._dec_samples, c=prob, cmap=cm, s=0.4, alpha=.75, zorder=1, label='Posterior sampling')

		fitted_position = ax.scatter(self._fit_coord[0], self._fit_coord[1], label='Fitted position', s=30,\
                marker='+', color='black', zorder=3)

		if self._real_coord != None:
                        real_position = ax.plot(self._real_coord[0], self._real_coord[1], linestyle='None', label='Real position', marker='*', 			      markersize=10, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=0.1, zorder=3)

                sun_coord = self._get_sun_position()
                        
                sun_position = ax.plot(sun_coord[0], sun_coord[1], linestyle='None', label='Sun position',\
                 marker='*', markersize=7, markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.1, zorder=3)
                        
		det_circles = []
		for name in self._det_list:
			det_center = self._detectors.detectors[name].get_center().transform_to('icrs')
			ra_circ = np.radians(-det_center.ra.value + 180.)
			dec_circ =  np.radians(det_center.dec.value)

			# fix overlapping between n5/b0 nb/b1 labels		
			text_pos = 'center'
			if 'n5' in self._det_list and 'b0' in self._det_list and name=='n5':
				text_pos = 'bottom'
			if 'n5' in self._det_list and 'b0' in self._det_list and name=='b0':
				text_pos = 'top'
			if 'nb' in self._det_list and 'b1' in self._det_list and name=='nb':
				text_pos = 'bottom'
			if 'nb' in self._det_list and 'b1' in self._det_list and name=='b1':
				text_pos = 'top'

			circ = CirclePolygon((ra_circ, dec_circ), np.radians(15), resolution=100, antialiased=True)
			ax.annotate(name, xy=(ra_circ, dec_circ), ha='center', va=text_pos, fontsize=10)
			det_circles.append(circ)

		det_coll = PatchCollection(det_circles, edgecolor=transp_black, linewidth=1.5, color=transp_green, zorder=1)
		ax.add_collection(det_coll)


		# ellisses axes
		a_50 = self._cl_50[0][1] - self._cl_50[0][0]
		b_50 = self._cl_50[1][1] - self._cl_50[1][0]

		a_90 = self._cl_90[0][1] - self._cl_90[0][0]
		b_90 = self._cl_90[1][1] - self._cl_90[1][0]

		t=np.linspace(0,2*np.pi,1000)
		ax.plot(self._fit_coord[0] + (a_50/2)*np.cos(t), self._fit_coord[1] + (b_50/2)*np.sin(t), linewidth=1.2, color='black')
		ax.plot(self._fit_coord[0] + (a_90/2)*np.cos(t), self._fit_coord[1] + (b_90/2)*np.sin(t), linewidth=1.2, color='black')

		ra_earth = self._earth_circle[0]
		dec_earth = self._earth_circle[1]
		earth_angular_radius = self._earth_circle[2]

		if (ra_earth+earth_angular_radius <= 2*np.pi) and (ra_earth-earth_angular_radius >= 0):
			earth_circ = CirclePolygon((ra_earth, dec_earth), earth_angular_radius, resolution=100,	antialiased=True, 						joinstyle='round', facecolor=transp_grey, edgecolor=transp_black, linewidth=1.5, zorder=0)
			ax.add_patch(earth_circ)
		else:
			earth_circ1 = CirclePolygon((ra_earth + np.pi, dec_earth), earth_angular_radius, resolution=100, antialiased=True, 					joinstyle='round', facecolor=transp_grey, edgecolor=transp_black, linewidth=1.5, zorder=0)
			ax.add_patch(earth_circ1)

			earth_circ2 = CirclePolygon((ra_earth - np.pi, dec_earth), earth_angular_radius, resolution=100, antialiased=True, 						joinstyle='round', facecolor=transp_grey, edgecolor=transp_black, linewidth=1.5, zorder=0)
			ax.add_patch(earth_circ2)


                
		ax.set_xticklabels(['330$^\circ$','300$^\circ$','270$^\circ$','240$^\circ$','210$^\circ$','180$^\circ$','150$^\circ$','120$^\circ$','90$^\circ$','60$^\circ$','30$^\circ$'])
		ax.grid()

		handles, labels = ax.get_legend_handles_labels()
		legend = ax.legend(handles, labels, loc='lower right')

		for i,handle in enumerate(legend.legendHandles):

			if labels[i]=='Posterior sampling':
				handle.set_sizes([20])		

		fig.savefig(self._save_path)


	def _get_gbm_geometry(self):

		pos_interpolator = gbmgeo.PositionInterpolator(T0=0,trigdat=self._trigdat_path)
		quaternion = pos_interpolator.quaternion(0)  # computed at t=0 (trigger time)
		dets = gbmgeo.GBM(quaternion)

		return dets


	def _get_earth_circle(self):
		
		trig_reader = TrigReader(self._trigdat_path, fine=False, verbose=False)
		trigger_time_pos = int(np.digitize([0.0],trig_reader._time_intervals.start_times)-1)
		dets = self._detectors
		#get detector position with respect to the earth frame
		dets.set_sc_pos(trig_reader._sc_pos[trigger_time_pos]*un.km)

		xyz_position = SkyCoord(x=dets._sc_pos[0],
                                y=dets._sc_pos[1],
                                z=dets._sc_pos[2],
                                frame='icrs',
                                representation='cartesian')

		ra_earth = xyz_position.transform_to('icrs').ra.value
		dec_earth = xyz_position.transform_to('icrs').dec.value

		#distance of the spacecraft from the center of the earth 
		df = dets._sc_pos.value
		#earth angular radius in radians
		earth_angular_radius = np.pi/2 - np.arccos(6371/np.sqrt((df**2).sum()) )

		return np.array([np.radians(-ra_earth + 180.), np.radians(dec_earth), earth_angular_radius])


	def _get_probability_array(self, pix):

		pixels=pix
		pixel_grid = np.full((pixels,pixels),0)

		min_ra = min(self._ra_samples)
		max_ra = max(self._ra_samples)

		min_dec = min(self._dec_samples)
		max_dec = max(self._dec_samples)


		for k in range (0, len(self._ra_samples)): 
			i1 = self._find_index(self._ra_samples[k], min_ra, max_ra, pixels)
			i2 = self._find_index(self._dec_samples[k], min_dec, max_dec, pixels)
			pixel_grid[i2][i1] += 1

		# normalize the pixel grid into a probability grid
		# each sampled point has an associated probability prob[k]

		prob=[]
		norm=0

		for k in range (0, len(self._ra_samples)):
			i1 = self._find_index(self._ra_samples[k], min_ra, max_ra, pixels)
			i2 = self._find_index(self._dec_samples[k], min_dec, max_dec, pixels)
			norm += pixel_grid[i2][i1]
			prob.append(float(pixel_grid[i2][i1]))
		
		prob = prob/norm		
		
		return np.array(prob)


	def _find_index(self, val, bot_lim, top_lim, pixels):
		# finds correct index in the pixels grid by splitting repeatedly the grid into two
		l=pixels
		delta = (top_lim - bot_lim)/l
		thr_index= l/2
		
		while l!=1:
		    l=l/2

		    if val>=bot_lim+(thr_index)*delta:

		        if l==1:
		            pass
		        else:
		            thr_index += l/2

		    elif val < bot_lim+(thr_index)*delta:

		        if l==1:
		            thr_index -= 1
		        else:
		            thr_index -= l/2
		            
		return int(thr_index)


        def _get_sun_position(self):

                trig_reader = TrigReader(self._trigdat_path, fine=False, verbose=False)
                
                time = Time(trig_reader.tobs, format='isot', scale='tt')
                sun_coord = get_sun(time)
                
                # GCRS ~ ICRS for the Sun
                return [-sun_coord.ra.rad+np.pi, sun_coord.dec.rad]
