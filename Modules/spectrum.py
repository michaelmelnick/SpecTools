import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utilities import *
from features import *

from scipy.optimize import curve_fit


class spectrum:
    """
    Spectrum holds an x,y dataset to model with a collection of features.
    
    
    
    """
    def __init__(self, fn, window=None, lineshape='gauss', name=None, Crunch=False):
        """
        fn is a valid filepath pointing to a csv on disk.
        the CSV is assumed to be a two column matrix 
        0th column is taken as the index
        
        fn can also be a pandas dataframe

        kwargs is a dictionary of keyword arguments to give read_csv
        """
        self.features = []
        self.fix_list = []
        self.covariance = []
        
        self.header, self.data = simple_data_parser(fn)
        
                    
        if name:
            self.name = name
        else:
            self.name = fn.split('.')[0]
        
        self.lineshape=lineshape
        self.look = np.array([True]*len(self.data[0]),dtype=bool)
        self.set_window(window)
        
        if Crunch:
            try:
                self.crunch()
            except:
                msg = "Didn't crunch {}".format(self.name)
                print(msg)
    
    @property            
    def x(self):
        """
        returns the x axis
        """
        return self.data[0][self.look]
    
    @property
    def y(self):
        """
        Returns the measure y axis
        """
        return self.data[1][self.look]
        
    @property
    def Rw(self):
        """
        Calculates Rw as estimate for goodness of fit.
        """
        y = self.y
        ycalc = self()
        Rw = np.sqrt(((y-ycalc)**2).sum()/(y**2).sum())
        
        return Rw
    
    @property
    def dataspacing(self):
        x = self.x
        dataspacing = (x.max()-x.min())/len(x)
        return dataspacing
                
    def __iter__(self):
        return self.features.__iter__()
    
    def __getitem__(self, idx):
        return self.features[idx]
        
    def __len__(self):
        return len(self.features)
    
    def __repr__(self):
        return self.name
        
    def __call__(self,x=None,*params):
        """
        Calling a spectrum returns the current y in the range
        corresponding to x. Passing new params
        changes the feature parameters.
        """
        
        params = list(params)
        
        for n, bol in enumerate(self.fix_list):
            if bol:
                try:
                    params.insert(n, self.fix_params[n])
                except IndexError:
                    params.append(self.fix_params[n]) 
        
        #How long should the parameter list be?    
        param_count = 0
        for feat in self.features:
            param_count += len(feat)
        
        #check the parameter list
        if not params:
            params = self.get_all_params()
        elif param_count != len(params):
            errString = "Need {} parameters, got {}" .format(self.param_count, len(params))
            raise ValueError(errString)
            
        #make sure we have an x-axis
        if x == None:
            x = self.x
        
        #parses the param list and pass to features.
        y = np.zeros_like(x)
        n = 0
        for feat in self.features:
            num_vals = len(ptypes[feat.ptype])
            t_params = params[n:n+num_vals]
            
            y += feat(x, *t_params)
            n += num_vals
            
        return y    
        
    def crunch_anypeaks(self,floor=0.1, lookahead=20, guesses = None):
        """
        Crunch finds peaks, fits them, sorts them and plots them. Guesses is a list of
        peak positions or features to use as fit starting places.
        
        Finds any number of peaks. Cannot fix peaks.
        """
        self.normalize_data()
        
        if guesses:
            
            guesstype = type(guesses[0])
            if guesstype == int or guesstype == float:
                self.peaks = self.guesses_to_peaks(guesses)
                self.define_features()
            elif guesstype == type(self):
                self.define_featureslike(guesses)
            else:
                raise TypeError('whoa something is wrong with your guesses!')
            
        else:
            self.find_peaks(lookahead=lookahead)
            self.define_features()
            
        self.refine_fit(floor=floor) 
        self.sort_features()
        self.plot_all()
        
    def crunch(self,floor=0.01, numpeaks = 2, normfactor = None, window = None, 
                lookahead = 20, guesses = None, extrapeaks = 0, fixed_peaks = {}, 
                minimum = None,loc=None):
        """
        Crunch: Finds peaks, fits them, and sorts them. Guesses is a list of
        peak positions or features to use as fit starting places. extrapeaks 
        allows one to add an arbitrary number of extra peaks to the guesses if 
        guesses is not a list of features. fixed_peaks is a dictionary of peaks 
        that one fixes in place of the form {peak number:[pname]}
        
        """
        self.set_window(window)
        self.normalize_data(normfactor, minimum,loc)
        
        if guesses:
            guesstype = type(guesses[0])
            if guesstype == int or guesstype == float:
                self.peaks = self.guesses_to_peaks(guesses)
                self.define_features()
                numpeaks = len(guesses) + extrapeaks
            elif guesstype == type(self):
                self.define_featureslike(guesses)
                numpeaks = len(guesses)
                
            else:
                raise TypeError('whoa something is wrong with your guesses!')
        
        else:
            self.find_peaks(lookahead=lookahead)
            self.define_features()

        self.refine_fit2(floor=floor, numpeaks=numpeaks, fix_dict = fixed_peaks)
        self.sort_features()
        
    def normalize_data(self, normfactor=None, minimum=None,loc=None):
        #Either save the data or reset the data
        if hasattr(self, 'data_raw'): 
            self.data = self.data_raw.copy()
        else:
            self.data_raw = self.data.copy()
        
        x = self.x
        y = self.y
        
        if not minimum:
            minimum = y[self.look].min()
        if not normfactor:
            normfactor = 1./(y[self.look].max() - minimum)
        
        if loc:
            idx = np.argmin((x-loc)**2)
            maximum = y[idx]
            
            normfactor = 1./(maximum - minimum)
            
        y -= minimum
        y *= normfactor
    
    
    
        
        
    def refine_fit(self,floor=0.01):
        self.fit_count = 0
        self._refine_fit(floor)
    
    
    def _refine_fit(self, floor=0.01):
        """
        Refine Fit iterates the fit while removing any feature less than the
        variance or with amplitude less than zero
        """
        
        print("Fitting {}".format(self.name))
        
        try:
            self.feature_fitter()
        except:
            pass
        
        try:
            self.fit_count += 1
        except:
            self.fit_count = 0
            
        if self.fit_count > 10:
            print('Too many calls to refine_fit')
            return
            
        variance = self.y.var()
                
        removable = []
        for feat in self.features:
            amp = feat['Amplitude']
            if amp < (variance*floor) or amp < 0.0:
                print('Removing a {} at {} for amplitude \'{}\' against value \'{}\''.format(feat.ptype, feat['Position'],amp,variance*floor))
                removable.append(feat)
        
        if removable:
            for kill in removable:
                try:
                    i = self.features.index(kill)
                    self.features.pop(i)
                    print('Removing a feature')
                except:
                    pass
            
            self._refine_fit(floor)
        
        else:
            diff = self() - self.y    
            if np.abs(diff.min()) > floor*variance:
            #if len(self.features) < numpeaks:
                
                loc = self.x[diff.argmin()]
                amp2 = self.y[diff.argmin()]
                width = self.dataspacing*10 #default peak width covers 10 points
                
                params = [amp2,width,loc]
            
                print('Adding a {} at {}'.format(self.lineshape, loc))
                
                self.add_feature(ptype=self.lineshape,params=params)
                self._refine_fit(floor)

    
    def refine_fit2(self,floor=0.01,numpeaks=2,fix_dict = {}):
        self.fit_count = 0
        self._refine_fit2(floor=floor,numpeaks=numpeaks,fix_dict=fix_dict) 
        
        
    def _refine_fit2(self, floor=0.01, numpeaks = 2, fix_dict = {}):
        """Refine Fit produces a fit with number of features defined by numpeaks."""
    
        print("Fitting {}".format(self.name))

        try:
            self.feature_fitter(fix_dict = fix_dict)
        except:
            pass
    
        try:
            self.fit_count += 1
        except:
            self.fit_count = 0
            
        if self.fit_count > 10:
            print('Too many calls to refine_fit')
            return
            
        variance = abs(self.y.var())
                
        removable = []
        for feat in self.features:
            amp = feat['Amplitude']
            if amp < (variance*floor) or amp < 0.0:
                print('Removing a {} at {} for amplitude \'{}\' against value \'{}\''.format(
                        feat.ptype, feat['Position'],amp,variance*floor))
                        
                removable.append(feat)

        else:
            if not removable and len(self.features) > numpeaks:
                removable.append(min(self.features, key = lambda feat: feat['Width']*feat['Amplitude']*np.sqrt(2*np.pi)))
                print('removing smallest')

        #either remove features (and redo the fit)
        #or add features (and redo the fit)
        #or finish
        
        if removable:
            for kill in removable:
                try:
                    i = self.features.index(kill)
                    self.features.pop(i)
                    print('Removing a feature')
                except:
                    pass
    
            self._refine_fit2(floor=floor,numpeaks=numpeaks,fix_dict=fix_dict)
        
        else:
            diff = self() - self.y
            #if np.abs(diff.min()/variance) > floor or len(self.features) < numpeaks:
            if len(self.features) < numpeaks:
            
                loc = self.x[diff.argmin()]
                amp2 = self.y[diff.argmin()]
                width = self.dataspacing*10 #default peak width covers 10 points
                
                params = [amp2,width,loc]
            
                print('Adding a {} at {}'.format(self.lineshape, loc))
                
                self.add_feature(ptype=self.lineshape,params=params)
                self._refine_fit2(floor=floor,numpeaks=numpeaks,fix_dict=fix_dict)         
    
    def set_window(self, window=None):
        """
        The window defines the section of the function to look at.
        """
        if window:
            self.window = window
        else:
            if not hasattr(self,'window'):
                x = self.x
                self.window = [x.min(), x.max()]    
        self.set_look()
    
    def set_look(self):
        """
        The look is a boolean array used to broadcast the section
        of the function to display or do math on.
        """
        if self.window:
            x = self.x
            self.look = (x >= min(self.window))*(x <= max(self.window))   
        else:
            self.set_window()
            
    
    def transform(self, xtransform=lambda x,y:x, ytransform=lambda x,y:y,xname=None,yname=None):
        """
        Transform allows for axis transformations.
        
        xtransform must be a function which transforms the x-axis, y must be a
        function to transform the y axis. Defaults to doing nothing.
        
        If xname or yname is passed, changes the headers for labeling purposes.
        Defaults to doing nothing.
        """
        
        x = self.x.copy()
        y = self.y.copy()
        
        self.x = xtransform(x,y)
        self.y = ytransform(x,y)
        
        window_min = xtransform(min(self.window),1)
        window_max = xtransform(max(self.window),1)
        self.set_window((window_min,window_max))
        
        if xname:
            self.header[0] = xname
            
        if yname:
            self.header[1] = yname
            
    def nm_to_eV(self,jacobian=True):
        """
        If the x-header is nm, coverts to eV. 
        If jacobian is true, applies the jacobian correction.
        """
        
        if self.header[0] == 'nm':
            
            xtransform = lambda x,y:hc/x
            
            if jacobian:
                ytransform = lambda x,y: (y*(x**2))/hc
            else:
                ytransform = lambda x,y: y
                
            self.transform(xtransform=xtransform,ytransform=ytransform,xname='eV')
            
        else:
            print('Warning, not in nm. No conversion applied. Use transform directly.')
            
    def eV_to_nm(self,jacobian=True):
        """
        If the x-header is eV, coverts to nm. 
        If jacobian is true, applies the jacobian correction.
        """
        
        if self.header[0] == 'eV':
            
            xtransform = lambda x,y:hc/x
            
            if jacobian:
                ytransform = lambda x,y: y*(x**2)/hc
            else:
                ytransform = lambda x,y: y
                
            self.transform(xtransform=xtransform,ytransform=ytransform,xname='nm')
        
        else:
            print('Warning, not in eV. No conversion applied. Use transform directly.')
                 
    
    
    def get_num_peaks(self):
        try:
            return self.num_peaks
        except:
            self.find_peaks()
            return self.num_peaks
    
    def get_peaks(self):
        try:
            return self.peaks
        except:
            self.find_peaks()
            return self.peaks
            
    def guesses_to_peaks(self, guesses):
        peaks = []
        xs = np.array(self.data.index, dtype = float)
        for guess in guesses:
            peaks.append(xs[np.argmin(np.abs(xs - guess))])
        return np.array(peaks)
               
    
    
        
    #Finding the peaks
    
    def find_peaks(self,mult=100,lookahead=20):
        self.smooth_data()
        self.take_derivatives()
        self.peak_detect(mult=mult, lookahead=lookahead)
        self.num_peaks = len(self.peaks)
    
    def smooth_data(self, win=11, order=2):
    
        y = self.y
        self.smoothed = savitzky_golay(y[self.look], win, order)
    
    def take_derivatives(self, win = 11, order = 2):
        try:
            y = self.y
            self.d1 = savitzky_golay(y[self.look], win, order, deriv=1)
            self.d2 = savitzky_golay(self.d1, win, order, deriv=1)
        except:
            raise
    
    def peak_detect(self, mult=100, lookahead=20):
    
        try:
            results = peakdetect(self.d2, x_axis = self.x,
                                lookahead=lookahead, delta = self.d2.var()*mult)
            
            self.peaks, self.peak_sig = np.array(zip(*results[1])[0]), np.array(zip(*results[1])[1])             
        
        except IndexError:
            self.peak_detect(mult=mult*.9, lookahead=lookahead)
                            
        except:
            if not hasattr(self, 'd2'):
                self.find_peaks()
            else:
                raise
        
    
    
        
                
    """
    Features are convenient ways to represent the components of a spectral model
    Currently they are either gaussians or scattering. This section is a work in 
    progress.
    
    This is the second attempt to model the spectrum as a collection of objects.
    The advantage of this method is that it can be generalized to many different
    functional forms with little effort. (hopefully)
    
    """
    
    
    
    
    def define_features(self,lineshape=None,width=10,add_bkgrnd=False):
        """
        Clears the feature list, creates a new list of gaussian features at each
        peak found by the second derivative method. It then adds one extra peak
        at the high energy edge of the window to model a gaussian background.
        
        width is the number of datapoints covered by a peak guess
        """
        
        if lineshape:
            self.lineshape = lineshape
        else:
            lineshape = self.lineshape
        
        self.clear_features()
        self.param_count = 0
        
        if not hasattr(self, 'peaks'):
            self.find_peaks()
        
        
        for peak in self.peaks:
            
            idx = ((self.x-peak)**2).argmin()
            
            params = [self.y[idx],self.dataspacing*width,peak]
            
            self.add_feature(ptype=lineshape,params=params)
    
            
    def add_feature(self,ptype,params,names=None,function=None):
        feat = feature(function=function,ptype=ptype,params=params,names=names)
        self.features.append(feat)
        
    def define_featureslike(self, featlist):
        self.features = []
        for feat in featlist:
            self.features.append(feature(ptype = feat.ptype, params = feat.params))
            
    def clear_features(self):
        self.features = []
        
    
    """
    Functions below attempt to fit the features defined using the methods above.
    """
    
    def get_all_params(self):
        """
        Extracts a list of parameters from all current features in the feature
        list. Useful for passing initialization parameters to curve_fit.
        """
        params = []
        for feat in self.features:
            for name in feat:
                params.append(feat[name])
        return params
        
    
    
    def sort_features(self):
        if self.features:
            self.features = sorted(self.features, key=lambda feat: feat['Position'])
    
    def feature_fitter(self, fix_dict={}, window=None):
        """
        The feature fitter attempts to find the best parameters to fit the spectrum
        with the features as defined. It uses the standard non-linear least squares
        method. 
        
        Outputs are stored in self.feat_fit and self.feat_curve. Returns the 
        parameters as a dictionary of dictionaries.
        
        fix_dict should be a dictionary of features to fix. An example is {1:['Position','Amplitude']}
        which would fix position and the amplitude of the first peak. The keys should
        either be integers referencing specific peaks or 'All' for all peaks.
        The values must name the parameter to fix using a valid parameter name.
        The fix_list builder will ingore invalid peak values and invalid param
        names.
        
        """
        #Change the window as requested for the fit
        if window:
            self.set_window(window=window)
        
        #make sure there are features to fit
        if not self.features:
            self.define_features()
        
        #Parse the fix list
        params = self.get_all_params()
        
        #build a list of parameters to fix based off the dictionary
        #fix dict
        fix_list = []
        pnames = []
                    
        for n, feat in enumerate(self.features):
            for pname in feat.names:
                pnames.append(pname)
                if n in fix_dict:
                    if pname in fix_dict[n]:
                        fix_list.append(True)
                    else:
                        fix_list.append(False)
                else:
                    fix_list.append(False)
        
        if 'All' in fix_dict:
            for n, pname in enumerate(pnames):
                if pname in fix_dict['All']:
                    fix_list[n] = True
        
        if len(fix_list) != len(params):
            raise ValueError('Fix_list is poorly built!')
        
        self.fix_params = params[:]
        self.fix_list = fix_list
        
        for n in range(len(fix_list)-1,-1,-1):
            if fix_list[n]:
                params.pop(n)
        
        
        x = self.x
        y = self.y
        self.feat_fit, self.feat_covariance = curve_fit(self, x, y, p0=params)
        
        #clean up the fix list so other functions work
        self.fix_list = []
        self.fix_params = []
        
        self.record_covariance()
        return self.get_feat_results()
    
    def record_covariance(self):
        n=0
        try:    
            cov = self.feat_covariance.diagonal()
        except:
            msg = 'covariance undefined'
            print(msg)
            return
        
        for feat in self.features:
            num_vals = ptypes[feat.ptype]
            feat.covariance = cov[n:n+num_vals]
        
                
    def get_feat_results(self):
        """
        Packs the feature parameter results into a Panda DataFrame.

        """
    
        feature_results = {}
        for n, feat in enumerate(self.features):
            temp = feat.get_results()
            temp['type'] = feat.ptype
            cur_peak = 'peak ' + str(n)
            feature_results[cur_peak] = temp
        self.feature_results = pd.DataFrame(feature_results)
        return self.feature_results
        
    def get_named_feat_results(self):
        """
        Packs the feature parameter results into a dictionary and returns.
        
        """
        feature_results = {}
        
        for n, feat in enumerate(self.features):
            temp = feat.get_named_results('Peak '+str(n))
            
            feature_results = dict(feature_results.items() + temp.items())
        
        return feature_results
        
    
    """
    The methods below are designed to plot the spectrum, the features, the model
    and the difference between the model and the spectrum for analysis purposes.
    """
    
    def plot_features(self, x=None, ax=None, kwargs={}, offset = 0.):
        """
        Plots the features on an axis.
        If x is undefined, x is taken as inside the window region.
        If ax is undefined, a new axis is generated
        Kwargs must be a dictionary of legal matplotlib keywords
        """
        
        if x == None:
            x = self.x
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        for feat in self.features:
            feat.plot(x, ax, kwargs, offset)
        
    def plot_spectrum(self, x=None, ax=None, kwargs={}, offset = 0., window = None):
        """
        Plots the original spectrum on an axis.
        If x is undefined, x is taken as inside the window region.
        If ax is undefined, a new axis is generated
        Kwargs must be a dictionary of legal matplotlib keywords
        """
        
        tempwindow = self.window
        
        if window:
            self.set_window([min(window), max(window)])
        
        if x == None:
            x = self.x
        
        #self.set_window([x.min(),x.max()])
        
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        kill_label = False
        if 'label' not in kwargs.keys():
            kwargs['label'] = self.name
            kill_label = True
        
        y = self.y + offset
        ax.plot(x,y,**kwargs)
        self.set_window(tempwindow)
        
        if kill_label:
            del kwargs['label']
        
        
    def plot_fit(self, x=None, ax=None, kwargs={}, offset = 0.):
        """
        Plots the composit result of the feature fit on an axis.
        If x is undefined, x is taken as inside the window region.
        If ax is undefined, a new axis is generated
        Kwargs must be a dictionary of legal matplotlib keywords
        """
        if x == None:
            x = self.x
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        kill_label = False
        if 'label' not in kwargs.keys():
            kwargs['label'] = 'Fit'
            kill_label = True
            
        y = self(x) + offset

        ax.plot(x,y,**kwargs)
        
        if kill_label:
            del kwargs['label']
        
    def plot_diff(self, x=None, ax=None, kwargs={}, offset = 0.):
        """
        Plots the difference between the feature fit and the original spectrum
        on an axis.
        If x is undefined, x is taken as inside the window region.
        If ax is undefined, a new axis is generated
        Kwargs must be a dictionary of legal matplotlib keywords
        """
        if x == None:
            x = self.x
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        kill_label = False
        if 'label' not in kwargs.keys():
            kwargs['label'] = 'Difference'
            kill_label = True
        

        y = self(x) + offset

        window = self.window
        self.set_window([x.min(),x.max()])
        
        y -= self.y
        ax.plot(x,y,**kwargs)
        plt.locator_params(axis = 'y', nbins = 4)
        self.set_window(window)
        
        if kill_label:
            del kwargs['label']
        
    def plot_all(self, x=None, kwargs={}):
        """
        Convenience function to plot everything on one axis.
        
        If x is undefined, x is taken as inside the window region.
        If ax is undefined, a new axis is generated
        Kwargs must be a dictionary of legal matplotlib keywords 
        """
        
        fig = plt.figure()
        ax1 = plt.subplot2grid((4,1),(0,0), rowspan=3)
        ax2 = plt.subplot2grid((4,1),(3,0))   
        self.plot_spectrum(x,ax1,kwargs)
        self.plot_features(x,ax1,kwargs)
        self.plot_fit(x,ax1,kwargs)
        self.plot_diff(x,ax2,kwargs)
        leg=ax1.legend(loc='best',fontsize='small')
        leg.get_frame().set_alpha(0.25)
        ax1.set_title(self.name)
        ax1.set_ylabel(self.header[1])
        ax2.set_ylabel('diff')
        ax2.set_xlabel(self.header[0])
        
        for ax in (ax1, ax2):
            ax.set_xlim(min(self.window),max(self.window))
        
        fig.set_size_inches(7,5) 
        fig.tight_layout()
        return fig
