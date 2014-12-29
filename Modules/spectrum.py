import pandas as pd
import numpy as np

from utilities import *
from features import feature

from scipy.optimize import curve_fit


class spectrum:
    """
    A spectrum is an x,y dataset with features which may be modeled as a set of gaussians.
    A spectrum may not be initialized empty.

    Currently Working:
    1) CSV files may currently be read from files on disk
    2) Peaks can be detected and modeled
    3) Results can be cleanly displayed from the spectrum
    4) Features are modeled in a modular manner
    
    """
    def __init__(self, fn, window=None, Crunch=False, index_col=0, lineshape='gauss', name=None, kwargs={}):
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
        
        kwargs['index_col'] = index_col
        if type(fn) == str:
            self.data = pd.read_csv(fn, **kwargs)
        else:
            self.data = fn
                    
        if name:
            self.name = name
        else:
            self.name = fn.split('.')[0]
        
        self.lineshape=lineshape
        
        self.key = self.data.keys()[0]
        self.define_axes()
        self.set_window(window)
        
        """
        #create Shortcut for special values
        for key in self.data.keys():
            setattr(self,key,self.data[key])
        """
        
        if Crunch:
            try:
                self.crunch()
            except:
                msg = "Didn't crunch {}".format(self.name)
                print msg
            
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
            
        
        self.fit_count = 0
        self.refine_fit(floor=floor) 
        self.fit_count = 0
        self.sort_features()
        self.plot_all()
        
    def crunch(self,floor=0.0001, numpeaks = 2, normfactor = None, window = None, lookahead = 20, guesses = None, extrapeaks = 0, fixed_peaks = {}, minimum = None, volume0 = None,loc=None):
        """Crunch: Finds peaks, fits them, and sorts them. Guesses is a list of
        peak positions or features to use as fit starting places. extrapeaks allows one to add an arbitrary number of extra peaks 
        to the guesses if guesses is not a list of features. fixed_peaks is a dictionary of peaks that one fixes in place of the 
        form {peak number:[pname]}"""
        self.set_window(window)
        
        if volume0:
            print "WARNING: Volume0 is depricated, use normfactor instead"
            normfactor = volume0/self.volume
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

        self.fit_count = 0
        self.refine_fit2(floor=floor, numpeaks=numpeaks, fix_dict = fixed_peaks) 
        self.fit_count = 0
        self.sort_features()
        
    def normalize_data(self, normfactor=None, minimum=None,loc=None):
        #Either save the data or reset the data
        if hasattr(self, 'data_raw'): 
            self.data = self.data_raw.copy()
        else:
            self.data_raw = self.data.copy()
        
        if not minimum:
            minimum = self.data[self.look].min()
        if not normfactor:
            normfactor = 1./(self.data[self.look].max() - minimum)
        
        if loc:
            x = np.array(self.data.index) #this is stupid
            y = self.data[self.data.keys()[0]].values
            idx = np.argmin((x-loc)**2)
            maximum = y[idx]
            
            normfactor = 1./(maximum - minimum)
            
            
            
            
            
        
        self.data = (self.data-minimum)*normfactor
    
    def normalize_data2(self, volume0, minimum):
        """For use with reaction class"""
        print "WARNING: normalize_data2 is depricated, use normalize_data"
        self.data_raw = self.data.copy()
        if not minimum:
            minimum = self.data[self.look].min()
        self.data = (self.data-minimum)*volume0/self.volume

    def refine_fit(self, floor=0.001):
        """
        Refine Fit iterates the fit while removing any feature less than the
        variance or with amplitude less than zero
        """
        
        print "Fitting {}".format(self.name)
        
        try:
            self.feature_fitter()
        except:
            pass
        
        try:
            self.fit_count += 1
        except:
            self.fit_count = 0
            
        if self.fit_count > 10:
            print 'Too many calls to refine_fit'
            return
            
        variance = abs(self.data[self.look][self.key].var())
                
        removable = []
        for feat in self.features:
            amp = feat.get_param('Amplitude')
            if amp < (variance*floor) or amp < 0.0:
                print 'Removing a {} at {} for amplitude \'{}\' against value \'{}\''.format(feat.ptype, feat.get_param('Position'),amp,variance*floor)
                removable.append(feat)
        
        #else:
        #    if not removable and len(self.features) > numpeaks:
        #        removable.append(min(self.features, key = lambda feat: feat['Width']*feat['Amplitude']*np.sqrt(2*np.pi)))
        #        print 'removing smallest'
        
        #either remove features (and redo the fit)
        #or add features (and redo the fit)
        #or finish
        
        if removable:
            for kill in removable:
                try:
                    i = self.features.index(kill)
                    self.features.pop(i)
                    print 'Removing a feature'
                except:
                    pass
            
            self.refine_fit(floor)
        
        else:
            diff = self.feature_model() - self.data[self.key][self.look].values    
            if np.abs(diff.min()/variance) > floor:
            #if len(self.features) < numpeaks:
                
                loc = self.data[self.look].index[diff.argmin()]
                amp2 = self.data[self.look][self.key][loc]
            
                print 'Adding a {} at {}'.format(self.lineshape, loc)
                getattr(self, 'add_'+self.lineshape)(loc, amp2)
                self.refine_fit(floor)

        
    def refine_fit2(self, floor=0.01, numpeaks = 2, fix_dict = {}):
        """Refine Fit produces a fit with number of features defined by numpeaks."""
    
        print "Fitting {}".format(self.name)

        try:
            self.feature_fitter(fix_dict = fix_dict)
        except:
            pass
    
        try:
            self.fit_count += 1
        except:
            self.fit_count = 0
            
        if self.fit_count > 20:
            print 'Too many calls to refine_fit'
            return
            
        variance = abs(self.data[self.look][self.key].var())
                
        removable = []
        for feat in self.features:
            amp = feat.get_param('Amplitude')
            if amp < (variance*floor) or amp < 0.0:
                print 'Removing a {} at {} for amplitude \'{}\' against value \'{}\''.format(feat.ptype, feat.get_param('Position'),amp,variance*floor)
                removable.append(feat)

        else:
            if not removable and len(self.features) > numpeaks:
                removable.append(min(self.features, key = lambda feat: feat['Width']*feat['Amplitude']*np.sqrt(2*np.pi)))
                print 'removing smallest'

        #either remove features (and redo the fit)
        #or add features (and redo the fit)
        #or finish
        
        if removable:
            for kill in removable:
                try:
                    i = self.features.index(kill)
                    self.features.pop(i)
                    print 'Removing a feature'
                except:
                    pass
    
            self.refine_fit2(floor, numpeaks)
        
        else:
            diff = self.feature_model() - self.data[self.key][self.look].values    
            #if np.abs(diff.min()/variance) > floor or len(self.features) < numpeaks:
            if len(self.features) < numpeaks:
            
                loc = self.data[self.look].index[diff.argmin()]
                amp2 = self.data[self.look][self.key][loc]
        
                print 'Adding a {} at {}'.format(self.lineshape, loc)
                getattr(self, 'add_'+self.lineshape)(loc, amp2)
                self.refine_fit2(floor, numpeaks)          
                            
    def add_note(self, note):
        """
        Add a note about spectral conditions for use in meta-analysis.
        Note must be a dictionary of notes.
        """
        for key in note:
            setattr(self, key, note[key])
    
    def set_window(self, window=None):
        """
        The window defines the section of the function to look at.
        """
        if window:
            self.window = window
        else:
            self.window = [self.data.index.values.min(), self.data.index.values.max()]    
        self.set_look()
    
    def set_look(self):
        """
        The look is a boolean array used to broadcast the section
        of the function to display or do math on.
        """
        if self.window:
            self.look = (self.data.index >= min(self.window))*(self.data.index <= max(self.window))   
        else:
            self.set_window()
    
    def reset_data(self, fn, window=None, sep=','):
        self.data = pd.reas_csv(fn,sep=sep, index_col=0)
        self.define_axes()
        self.set_window(window) 
    
    def define_axes(self):
        """
        Makes sure the x-axis is in eV and converts it if necessary
        """
        if self.data.index.name == 'nm':
            #If x-axis is in nm, convert to eV
            self.x_nm = np.array(self.data.index.tolist())
            self.x_eV = hc/self.x_nm
            self.data.index = self.x_eV
            self.data.index.name = 'eV'
            
            #Apply Jacobian correction - not doing this!
            #See dx.doi.org/10.1021/jz401508t
            #self.data = self.data / (hc/(self.x_eV**2))   
            
        elif self.data.index.name == 'eV':
            self.x_eV = np.array(self.data.index.tolist())
            self.x_nm = hc/self.x_eV
            
        elif self.data.index.name == 'ppm':
            self.x_ppm = np.array(self.data.index.tolist())
            
        else:
            raise TypeError("Malformed X-axis in raw data.")
            
        self.values = self.data[self.key].values
    
    
    def get_window(self):
        return self.window
        
    def get_look(self):
        return self.look
                 
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
        self.smoothed = savitzky_golay(self.data[self.key][self.look].values, win, order)
    
    def take_derivatives(self, win = 11, order = 2):
        try:
            self.d1 = savitzky_golay(self.data[self.key][self.look].values, win, order, deriv=1)
            self.d2 = savitzky_golay(self.d1, win, order, deriv=1)
        except:
            raise
    
    def peak_detect(self, mult=100, lookahead=20):
        
        xs = np.array(self.data.index.tolist())[self.look]
        try:
            results = peakdetect(self.d2, x_axis = xs,
                                lookahead=lookahead, delta = self.d2.var()*mult)
            
            self.peaks, self.peak_sig = np.array(zip(*results[1])[0]), np.array(zip(*results[1])[1])             
        
        except IndexError:
            self.peak_detect(mult=mult*.9, lookahead=lookahead)
                            
        except:
            if not hasattr(self, 'd2'):
                self.find_peaks()
            else:
                raise
                
        #questionable idea:
        #throw out all peaks less than the variance of the signal.
        #this might cause problems in some applications
        
        #test = np.abs(self.data[self.key][self.peaks].values) > abs(self.data[self.look][self.key].var())
        #self.peaks = self.peaks[test]
        
    def __iter__(self):
        return self.features.__iter__()
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def __repr__(self):
        return self.name
    
        
                
    """
    Features are convenient ways to represent the components of a spectral model
    Currently they are either gaussians or scattering. This section is a work in 
    progress.
    
    This is the second attempt to model the spectrum as a collection of objects.
    The advantage of this method is that it can be generalized to many different
    functional forms with little effort. (hopefully)
    
    """
    
    def define_features(self,lineshape=None,width=0.02,add_bkgrnd=False):
        """
        Clears the feature list, creates a new list of gaussian features at each
        peak found by the second derivative method. It then adds one extra peak
        at the high energy edge of the window to model a gaussian background.
        """
        
        if lineshape:
            self.lineshape = lineshape
        else:
            lineshape = self.lineshape
        
        self.features = []
        self.param_count = 0
        
        if not hasattr(self, 'peaks'):
            self.find_peaks()
        
        for peak in self.peaks:
            
            params = {  
                        'loc': peak,
                        'amp': self.data[self.key][peak],
                        'width': width
                     }
            
            getattr(self,'add_'+lineshape)(**params)
        
        if add_bkgrnd:
            self.add_bkgrnd_gauss(width)
    
    def define_featureslike(self, featlist):
        self.features = []
        for feat in featlist:
            self.features.append(feature(ptype = feat.ptype, params = feat.params))
    
    def add_gauss(self, loc, amp, width=0.02):
        """
        Add a gaussian to the feature list with location and amplitude given.
        Width may be passed as a keyword argument. Default is 20 meV
        """
        params = [amp, width, loc]
        ptype = 'gauss'
        self.features.append(feature(ptype, params))
        

    def add_bkgrnd_gauss(self,width=0.02):
        """
        Adds a background gaussian at the highest energy end of the window.
        The advantage of this method is that it carries the 'bkgrnd' label
        for plotting and tabulating results.
        """
        
        params = []
        params.append(self.data[self.key][self.look][-1])
        params.append(0.02)
        params.append(self.data[self.look].index.values.max())
        
        self.features.append(feature(ptype='bkgrnd_gauss', params=params))
    
    def add_lorentz(self, loc, amp, width=0.2):
        """
        Add a lorentzian to the feature list with location and amplitude given.
        Width may be passed as a keywork argument. Default is 0.2ppm
        """
        params = [amp, width, loc]
        ptype = 'lorentz'
        self.features.append(feature(ptype, params, names[ptype]))
        
    
    def add_scatter(self, a):
        """
        Adds an y=a/x**4 dependent background feature. Might be useful in cases
        with large ammounts of scattering.
        """
        
        self.features.append(feature(ptype='scatter', params=[a]))
        
    def add_flat(self, a=0):
        """
        Adds a flat baseline with guess value 'a'
        """
        self.features.append(feature(ptype='flat', params=[a]))
        
    def add_line(self, m, b):
        """
        Adds a strait line with slope 'm' and intercept 'b'
        """
        self.features.append(feature(ptype='line', params=[m, b]))
        
    
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
            for p in feat.params:
                params.append(p)
        return params
        
    def feature_model(self, x=None, *params):
        """
        This generates a set of y values for a given set of x values using a list
        of parameters passed. If params is None, uses the previously defined params.
        
        If x is passed empty, the x-parameters will be generated from the current
        window.
        
        All params pased will change the values in the features. If used with
        curve_fit the resulting feature parameters will be the last parameters
        passed.
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
            param_count += ptypes[feat.ptype]
        
        #check the parameter list
        if not params:
            params = self.get_all_params()
        elif param_count != len(params):
            errString = "Need {} parameters, got {}" .format(self.param_count, len(params))
            raise ValueError(errString)
            
        #make sure we have an x-axis
        if x == None:
            x = np.array(self.data[self.look].index.tolist())
        
        #parses the param list and pass to features.
        y = np.zeros_like(x)
        n = 0
        for feat in self.features:
            num_vals = ptypes[feat.ptype]
            t_params = params[n:n+num_vals]
            
            y += feat.model(x, params=t_params)
            n += num_vals
            
        return y
    
    def sort_features(self):
        if self.features:
            self.features = sorted(self.features, key=lambda feat: feat.get_param('Position'))
    
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
        
        
        x = np.array(self.data[self.look].index.tolist())
        y = self.data[self.key][self.look].values
        self.feat_fit, self.feat_covariance = curve_fit(self.feature_model, x, y, p0=params)
        
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
            print msg
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
            x =np.array(self.data[self.look].index.tolist())
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
            x = np.array(self.data[self.look].index.tolist())
        
        self.set_window([x.min(),x.max()])
        
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        kill_label = False
        if 'label' not in kwargs.keys():
            kwargs['label'] = self.name
            kill_label = True
        
        y = self.data[self.key][self.look].values + offset
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
            x = np.array(self.data[self.look].index.tolist())
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        kill_label = False
        if 'label' not in kwargs.keys():
            kwargs['label'] = 'Fit'
            kill_label = True
            
        y = self.feature_model(x) + offset

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
            x =np.array(self.data[self.look].index.tolist())
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        kill_label = False
        if 'label' not in kwargs.keys():
            kwargs['label'] = 'Difference'
            kill_label = True
        

        y = self.feature_model(x) + offset

        window = self.window
        self.set_window([x.min(),x.max()])
        
        y -= self.data[self.key][self.look].values
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
        ax1.set_ylabel(self.key)
        ax2.set_ylabel('diff')
        ax2.set_xlabel(self.data.index.name)
        
        for ax in (ax1, ax2):
            ax.set_xlim(min(self.window),max(self.window))
        
        fig.set_size_inches(7,5) 
        fig.tight_layout()
        return fig
