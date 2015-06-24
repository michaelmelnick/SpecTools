import pandas as pd
import numpy as np

from spectrum import Spectrum
from features import Feature
from utilities import *

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from os.path import join
from os import listdir



class Reaction():
#   init
#   sorting
#   fit
#   plotting results
#   plotting spectra
#   pass
    
    def __iter__(self):
        for key in sorted(self.spectra):
            yield key
    
    def __getitem__(self, key):
        return self.spectra[key]
        
    def __len__(self):
        return len(self.spectra)
        
    def keys(self):
        return sorted(self.spectra.keys())
        
    def values(self):
        for key in self:
            yield self[key]
       
    def __init__(self,directory, **kwargs):

        """loads each Spectrum in a particular Reaction and creates a dictionary of spectra, 
        which can then be passed to fit_reaction.
        
        Args:
        Directory - a directory containing data files. All datafiles must have the same
        extension, default '.csv'
        
        Reaction initialization can be modified by several keyword arguments.
        Kwargs:
        
        window - a list or tuple containing the minimum and maximum range to fit
        the data.
        
        extension - a string to test for to include files in the load order. Default
        is '.csv'.
        
        spec_kwargs - a dictionary of keywords arguments to pass to Spectrum during
        file loading. Default {'skiprows':1,'usecols':[0,1]}
        
        file_namer - A function, which if passed a filename as a string, returns
        a value to use as a Spectrum label.
        
        Setting the X axis:
            
        x_name - The name of the Reaction axis, default is 'Number'
        
        Choose one of the following, defaults to numbering the spectra
        x_values - a list or array of values
        
        x_function - a function which takes the number of reactions and returns
        a list or array of values
        
        x_file - a filepath which contains the x_values
        
        """

        self.spectra = {}
        self.__kwargs = kwargs
        
        if 'window' in kwargs:
            self.window = kwargs['window']
        else:
            self.window = (np.inf,-np.inf)
            
        #load data
        self.directory = directory
        self.load_data(**kwargs)
        self.make_x_axis(**kwargs)
        self.set_normfactors(**kwargs)
        
    def load_data(self,**kwargs):
        """
        Loads all the data in the directory specified. Accepts keyword arguments
        
        spec_kwargs : keyword arguments to pass to Spectrum
           defaults to: skiprows=1,usecols=[0,1] 
        
        file_namer : a function which converts a filename string to a name
        cycletime : if passed, file namer takes the cycle number
        addedtitle : if passed, takes the value between the first and second periods
        as the file name
        
        if no namer information is passed, defaults to taking the section of the
        string before the first period
        
        """
        
        directory = self.directory
        
        if 'extension' in kwargs:
            extension = kwargs['extension']
        else:
            extension = '.csv'
        
        fl = [join(directory,fn) for fn in listdir(self.directory) if extension in fn]
        #Loading the data
        if 'file_namer' in kwargs:
            file_namer = kwargs['file_namer']
            
        elif 'cycletime' in kwargs:
            file_namer = lambda name: int(name.split('.')[-2].strip('Cycle'))
            
        elif 'addedtitles' in kwargs:
            file_namer = lambda name: name.split('.')[1]
            
        else:
            file_namer = lambda name: name.split('.')[0]
        
        
            
        self.spectra = {file_namer(fn):Spectrum(fn,name=file_namer(fn)) for fn in fl}
        return
        
    def set_windows(self,window=None):
        if window:
            self.window = window
        else:
            window = self.window
        
        for key in self:
            self[key].set_window(window)
            
    
    def make_x_axis(self,x_name=None,x_method=None, **kwargs):
        #make x_axis
        self.__x_methods = [    
                            'x_values',
                            'x_function',
                            'x_file'
                            ]
                        
        
        #Name the spectral axis
        if 'x_name':
            self.x_name = x_name
        else:
            self.x_name = 'Number'
        
            #
        if 'x_values' == x_method or 'x_values' in kwargs:
            self.x_values = np.array(kwargs['x_values'])
            
        elif 'x_function' == x_method:
            func = kwargs['x_function']
            if 'x_func_kwargs' in kwargs:
                fkwargs = kwargs['x_func_kwargs']
                self.x_values = func(len(self),**fkwargs)
            else:
                self.x_values = np.array(func(len(self)))
        
        elif 'x_file' == x_method:
            fn = kwargs['x_file']
            if 'x_reading_kwargs' in kwargs:
                rkwargs = kwargs['x_reading_kwargs']
                self.x_values = np.loadtxt(fn, **rkwargs)
            else:
                self.x_values = np.loadtxt(fn)
                
        else:
            self.x_values = np.arange(1,len(self)+1)
            
        
        if len(self) != len(self.x_values):
            print "Warning: Reaction x_values are wrong length, defaulting to number axis"
            self.x_values = np.arange(1,len(self)+1)
            
        return
            
    def set_normfactors(self,normfactors=None,**kwargs):
        if normfactors:
            self.normfactors = normfactors
            if len(normfactors) == len(self):
                return    

        self.normfactors = [None]*len(self)
        return
    
    
    def find_minimum(self, window = None):
        self.minimum = np.Infinity
        if window:
            self.set_windows(window)
        for key in self:
            spec = self[key]
            if min(spec.values[spec.look]) < self.minimum:
                self.minimum = min(spec.values[spec.look])
        

    def fit_reaction(self, outfile=None, numpeaks = 2, floor = 0.01, window = None, 
    lookahead=20, guesses = None, extrapeaks = 0, informnext = False, fix_last = False,find_min=False,reverse=False,loc=None,**kwargs):
        """
        
        takes a Reaction dictionary and returns fits for each instance in the dictionary
        in the form of a panda DataFrame, which can be exported as a .csv named fn.
    
        if the Features are gaussians, then the width param is converted into fwhm, and the 
        area of the gaussian is also computed and included in the output.
        
        guesses are a list of expected peak positions. the length of the list defines the number of peaks to be fit
        unless extra peaks are wanted and extrapeaks is passed a value."""
    
        self.results = results = {}
        fixed_peaks = {}
        if find_min:
            self.find_minimum(window = window)
        else:
            self.minimum=None
            
        if reverse:
            keys = self.keys()[::-1]
        else:
            keys = self.keys()        
                        
        for n, key in enumerate(keys):
            spec = self[key]
            spec.crunch(floor = floor, numpeaks=numpeaks, normfactor = self.normfactors[n],
            window = window, lookahead = lookahead, guesses = guesses, extrapeaks = extrapeaks, fixed_peaks = fixed_peaks,
            minimum = self.minimum,loc=loc)  
            results[key] = spec.get_named_feat_results()
            if informnext:
                guesses = spec.Features
            if fix_last:
                fixed_peaks = {numpeaks - 1: ['Position', 'Amplitude', 'Width']}
                
        self.tabulate_results(outfile)
        return

    
    def tabulate_results(self,outfile=None):
        '''
        tabulate_results iterates over the Reaction and extracts the fit parameters
        and tabulates them into a panda dataframe. The frame can be exported to a valid
        outfile filepath if passed as a string.
        '''
        
        self.results = results = {}
        for key in self:
            spec = self[key]
            results[key] = spec.get_named_feat_results()
        
        
        self.results = results = pd.DataFrame(results).transpose()
    
        for key in self.spectra.keys():
            if not self.spectra[key].lineshape == 'gauss':
                break
        else:
            for n in range(len(results.keys())/3):
                amp_key = 'Peak '+str(n)+' Amplitude'
                wid_key = 'Peak '+str(n)+' Width'
                amps = results[amp_key]
                wids = results[wid_key]
                results.insert(len(results.keys()),'Peak '+str(n)+' FWHM', 2.35482*abs(wids)) 
                results.insert(len(results.keys()),'Peak '+str(n)+' Area', amps*abs(wids)*np.sqrt(2*np.pi))
            results.sort(axis = 1, inplace = True)
            
        results.index = self.x_values
        results.index.name = self.x_name

        if outfile:
            results.to_csv(outfile)
    
        return results



    def plot_fitresults(self, export = False):
        """returns a list of figures and axes of the fitted parameters"""
        if hasattr(self, 'results'):
            figs = []
            axes = []
            results = self.results
            xs = results.index
            for key in results:
                ys = results[key].values
                if 'Amplitude' in key:
                    yaxlab = 'Amplitude (a.u.)'
                elif 'Area' in key:
                    yaxlab = 'Peak area'
                elif 'FWHM' in key:
                    yaxlab = 'FWHM (eV)'
                elif 'Position' in key:
                    yaxlab = 'Position (eV)'
                elif 'Width' in key:
                    yaxlab = 'Width (eV)'
                else:
                    print 'Err, keys are broken.'
                figs.append(plt.figure())
                plt.plot(xs, ys, 'bo', label = key)
                axes.append(plt.gca())
                axes[-1].set_xlabel(results.index.name)
                axes[-1].set_ylabel(yaxlab)
                axes[-1].set_title(self.directory.split('/')[-1]+ ' ' + key)
                if export:
                    figs[-1].savefig(self.directory.split('/')[-1]+ ' ' + key + '.svg')
                    figs[-1].savefig(self.directory.split('/')[-1]+ ' ' + key + '.png')
            return figs, axes
        else:
            print 'This Reaction has not yet been fitted or something else is wrong.'
    
        
    def plot_fitresultssbs(self, export = False, map_name = 'winter', feat_list = None):
        """returns plots of the fitted parameters, side by side. feat_list is a list of Features that one wants to plot."""

        if hasattr(self, 'results'):
            
            c_map = plt.get_cmap(map_name)
            cNorm = colors.Normalize(vmin=0, vmax=len(self.results))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=c_map)
            
            axes = []
            self.resultfig = fig = plt.figure()
            results = self.results.copy()
            if feat_list:
                del_list = range(len(self.results.keys())/5)
                for feat in feat_list:
                    del_list.remove(feat)
                for key in results:
                    for feat in del_list:
                        if str(feat) in key:
                            del results[key]
            for key in results:
                if 'Width' in key:
                    del results[key]
            numrows = range(len(results.keys())/4)
            numcols = range(4)
            
            for i in numrows:
                axes.append([])
                rowaxes = axes[-1]
                
                for j in numcols:
                    rowaxes.append(plt.subplot2grid((len(numrows), len(numcols)), (i,j)))
                    ax = rowaxes[-1]
                    key = results.keys()[i*len(numcols)+j]
                    col = results[key]
                    
                    if 'Amplitude' in key:
                        yaxlab = 'Amplitude (a.u.)'
                    elif 'Area' in key:
                        yaxlab = 'Peak area'
                    elif 'FWHM' in key:
                        yaxlab = 'FWHM (eV)'
                    elif 'Position' in key:
                        yaxlab = 'Position (eV)'
                    else:
                        print 'Err, keys are broken.'
                    
                    ax.set_ylabel(yaxlab)
                    ax.set_xlabel(results.index.name)
                    ax.set_title(key)
                   
                    for n in range(len(col)):
                        colorVal = scalarMap.to_rgba(n)
                        #kwargs['color'] = colorVal
                        ax.plot(col.index[n], col.values[n], 'o', color = colorVal)
                    
                    #ran = [min(ax.get_yticks()), max(ax.get_yticks())]
                    #numticks = 5
                    #ticklist = [n * (ran[1]-ran[0])/(float(numticks)-1.) for n in range(numticks)]
                    #ax.set_yticks(ticklist)
                    plt.setp(ax.get_xticklabels()[::2], visible=False)
                    plt.rc('xtick', labelsize = 10)
                    plt.rc('ytick', labelsize = 10)
            
            fig.set_size_inches(10.5,3.5*len(axes))
            fig.tight_layout()
            
            if export:
                    fig.savefig(self.directory + '/../' + self.directory.split('/')[-1]+ ' fitting results' + '.svg')
                    fig.savefig(self.directory + '/../' + self.directory.split('/')[-1]+ ' fitting results' + '.png')
            
            return fig
        
        else:
            print 'This Reaction has not yet been fitted or something else is wrong.'


    def plot_fitresultscollapsed(self, export = False, map_list = None, feat_list = None):
        """returns plots of the fitted parameters with each Feature plotted on the same figure. map_list is a list of colormaps 
        differentiating the color of each Feature. if None, default colormaps list will be used. 
        feat_list is a list of Features that one wants to plot."""

        if hasattr(self, 'results'):
            
            if not map_list:
                map_list = ['winter', 'autumn', 'cool', 'afmhot', 'autumn', 'bone', 'copper',
                             'gist_heat', 'gray', 'hot', 'pink', 'summer']
            
            
            #c_map = plt.get_cmap(map_list[n])
            #cNorm = colors.Normalize(vmin=0, vmax=len(self.results))
            #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=c_map)
            
            axes = []
            self.resultfig = fig = plt.figure()
            results = self.results.copy()
            
            if feat_list:
                del_list = range(len(self.results.keys())/5)
                for feat in feat_list:
                    del_list.remove(feat)
                for key in results:
                    for feat in del_list:
                        if str(feat) in key:
                            del results[key]
            for key in results:
                if 'Width' in key:
                    del results[key]
            numpeaks = range(len(results.keys())/4)
            numparams = range(4)

                
            for j in numparams:
                axes.append(plt.subplot2grid((1, len(numparams)), (0,j)))
                ax = axes[-1]
                #legend = []
                
                for i in numpeaks:
                    key = results.keys()[i*len(numparams)+j]
                    col = results[key]
                    
                    #legend.append(str(i) + ' = ' + map_list[i])
                    c_map = plt.get_cmap(map_list[i])
                    cNorm = colors.Normalize(vmin=0, vmax=len(self.results))
                    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=c_map)
                        
                    for n in range(len(col)):
                        colorVal = scalarMap.to_rgba(n)
                        ax.plot(col.index[n], col.values[n], 'o', color = colorVal)
                
                if 'Amplitude' in key:
                    yaxlab = 'Amplitude (a.u.)'
                elif 'Area' in key:
                    yaxlab = 'Peak area'
                elif 'FWHM' in key:
                    yaxlab = 'FWHM (eV)'
                elif 'Position' in key:
                    yaxlab = 'Position (eV)'
                else:
                    print 'Err, keys are broken.'
                
                ax.set_ylabel(yaxlab)
                ax.set_xlabel(results.index.name)
                #ax.set_title(key)
                    
                plt.setp(ax.get_xticklabels()[::2], visible=False)
                plt.rc('xtick', labelsize = 10)
                plt.rc('ytick', labelsize = 10)
            
            fig.set_size_inches(14,3.5)
            fig.tight_layout()
            
            if export:
                    fig.savefig(self.directory + '/../' + self.directory.split('/')[-1]+ ' fitting results' + '.svg')
                    fig.savefig(self.directory + '/../' + self.directory.split('/')[-1]+ ' fitting results' + '.png')
            
            return fig
        
        else:
            print 'This Reaction has not yet been fitted or something else is wrong.'


    def plot_spectra(self, subset = None, x = None, window = None):
        """returns individual plots of a subset of spectra in a given Reaction"""
        if not subset:
            subset = [0, len(self.spectra)]
        
        if window:
            self.set_windows(window)
        else:
            window = self.window
        
        figs = []
        axes = []
        for n in range(*subset):
            figs.append(plt.figure())
            ax = plt.gca()
            axes.append(ax)
            spec = self.spectra[sorted(self.spectra.keys())[n]]
            spec.plot_spectrum(x, axes[-1], window = window)
            ax.set_ylabel('Absorbance (a.u.)')
            ax.set_xlabel('eV')
    
        return figs, axes


    def plot_spectrasbs(self, subset = None, offset = 0., x = None, window = None, map_name = 'winter', kwargs = {}):
        """returns individual plots of a subset of spectra in a given Reaction"""
        if not subset:
            subset = [0, len(self.spectra)]
        
        if window:
            self.set_windows(window)
        else:
            window = self.window
        
        fig = plt.figure()
        ax = fig.gca()
        c_map = plt.get_cmap(map_name)
        cNorm = colors.Normalize(vmin=0, vmax=subset[1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=c_map)
        
        for n in range(*subset):
            colorVal = scalarMap.to_rgba(n)
            kwargs['color'] = colorVal
            spec = self.spectra[sorted(self.spectra.keys())[n]]
            spec.plot_spectrum(x, ax, kwargs, offset*n, window = window)
        
        ax.set_ylabel('Absorbance (a.u.)')
        ax.set_xlabel('eV')
        ax.set_title(self.directory.split('/')[-1])
        
        if window:
            ax.set_xlim(min(window), max(window))
        
        return fig, ax
    
    def plot_spectrafits(self, subset = None):
        """returns individual plots of a subset of fitted spectra in a given Reaction"""
        if hasattr(self, 'results'):
            if not subset:
                subset = [0, len(self.spectra)]
            
            figs = []
            axes = []
            for n in range(*subset):
                figs.append(self.spectra[sorted(self.spectra.keys())[n]].plot_all())
                axes.append(plt.gca())
            return figs, axes
        else:
            print 'This Reaction has not yet been fitted or something else is wrong.'
    
    def plot_spectrafitssbs(self, x=None, offset = 0., subset = None, map_name = 'winter', export = False, kwargs = {}):
        """returns plots of a subset of fitted spectra in a given Reaction in a single figure"""
        if hasattr(self, 'results'):
            if not subset:
                subset = [0, len(self.spectra)]
            
            #axis_font = {'fontname':'Arial', 'size':'14'}
            c_map = plt.get_cmap(map_name)
            cNorm = colors.Normalize(vmin=0, vmax=subset[1])
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=c_map)
            
            fig = plt.figure()
            ax1 = plt.subplot2grid((7,2),(0,0), rowspan = 4)
            ax2 = plt.subplot2grid((7,2),(0,1), rowspan = 4)
            ax3 = plt.subplot2grid((7,2),(5,0), rowspan = 2)
            ax4 = plt.subplot2grid((7,2),(5,1), rowspan = 2)
            ax4.get_yaxis().tick_right()
            ax4.yaxis.set_label_position('right')
            ax4.yaxis.set_ticks_position('both')
            for n in range(*subset):
                colorVal = scalarMap.to_rgba(n)
                kwargs['color'] = colorVal
                spec = self.spectra[self.keys()[n]]
                spec.plot_spectrum(x, ax1, kwargs, offset*n)
                spec.plot_Features(x, ax3, kwargs, offset*n)
                spec.plot_fit(x, ax2, kwargs, offset*n)
                spec.plot_diff(x, ax4, kwargs, offset*n)
                
            ax1.set_title('Originals')
            ax1.set_ylabel('a.u.')
            ax2.set_title('Fits')
            ax3.set_title('Features')
            ax3.set_ylabel('a.u.')
            ax3.set_xlabel('eV')
            ax4.set_title('Residuals')
            ax4.set_ylabel('diff')
            ax4.set_xlabel('eV')
                
            
            for ax in (ax1, ax2, ax3, ax4):
                ax.set_xlim(min(spec.window),max(spec.window))
                
            ax2.set_ylim(ax1.get_ylim())
            
            if export:
                tstamp = timestamp()
                fig.savefig(self.directory.split('/')[-1]+ ' ' + tstamp + '.svg')
                fig.savefig(self.directory.split('/')[-1]+ ' ' + tstamp + '.png')
                
            fig.set_size_inches(7,5)
            return fig
        else:
            print 'This Reaction has not yet been fitted or something else is wrong.'
        #fig.set_size_inches(8,10.5) 
        #fig.tight_layout()
        
    def remove_spectrum(self, key):
        specnum = sorted(self.spectra.keys()).index(key)
        self.spectra.pop(key)
        self.dilutions = np.delete(self.dilutions, specnum)
        if hasattr(self, 'molarities'):
            self.molarities = np.delete(self.molarities, specnum)
        if hasattr(self, 'times'):
            self.times = np.delete(self.times)
