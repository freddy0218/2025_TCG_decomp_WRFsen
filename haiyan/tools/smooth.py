import numpy as np
def boxcar(A, nodata, window_size=3):
	"""
	https://stackoverflow.com/questions/57097684/apply-boxcar-average-to-geospatial-image
	"""
	from scipy.signal import fftconvolve
	mask = (A==nodata)
	K = np.ones((window_size, window_size),dtype=int)
	out = np.round(fftconvolve(np.where(mask,0,A), K, mode="same")/fftconvolve(~mask,K, mode="same"), 2)
	out[mask] = nodata
	return np.ma.masked_array(out, mask=(out == nodata))
