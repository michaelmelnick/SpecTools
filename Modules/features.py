#This dictionary connects the known types of 
#spectral features to the number of parameters
#necessary to define that feature.

import numpy as np
import matplotlib.pyplot as plt
from utilities import *

from inspect import getargspec


def _gauss(x,A,sigma,x0):
    return A*np.exp(-(x-x0)**2/((2.*sigma)**2))
            
def _lorentz(x,A,gamma,x0):
    return A*(gamma**2)/((x-x0)**2 + gamma**2)
    
def _scatter(x,A,B): 
    return (A/x**4) + (B/x**2)

def _flat(x,A): 
    return A*np.ones(len(x))

def _line(x, m, b):
    return m*x + b

def _sqrt(x,A,x0):
    return A*np.sqrt(x-x0)
    
def _none(x):
    return np.zeros_like(x)
    
def _multiplet(x,order,coupling,A,gamma,x0):
    y = np.zeros_like(x)
    fib = binomial_row(order-1)
    total = sum(fib)
    for n, val in enumerate(fib):
        Amp = A*float(val)/total
        pos = float(x0) + coupling*n - coupling*(order-1)/2
        
        y+= _lorentz(x,Amp,gamma,pos)
        
    return y

ptypes = {  'gauss': ['Amplitude', 'Width', 'Position'],
            'lorentz': ['Amplitude', 'Width', 'Position'],
            'scatter': ['A','B'],
            'flat':['Baseline'],
            'line':['Slope','Intercept'],
            'None':[],
            'sqrt':['Amplitude','BandGap'],
            'multiplet':['Order','Coupling','Amplitude','Width','Position']   
         }
         
funcs = {
            'gauss': _gauss,
            'lorentz': _lorentz,
            'scatter': _scatter,
            'flat': _flat,
            'line': _line,
            'None': _none,
            'sqrt': _sqrt,
            'multiplet': _multiplet
        }
            
class feature:
    """
    A feature is the basic object in a spectrum. It consists of a type name,
    a function, a list of parameters, and a list of names. When called with an
    array of values, the feature returns the results of passing those values
    to the function along with the stored parameters. If new parameters are passed
    along with the call, the parameter list is updated before evaluating the
    function. If initialized empty, returns an array of zeros the same length
    as the original array.
    
    Kwargs:
    
    function: an object with a __call__ method. Must accept one array argument.
              For best results it should return an array of the same length.
              
    ptype: A string naming the feature. If ptype is a key in ptypes and funcs, the
            feature will be automatically populated with a function and names
    
    params: list of positional parameters to initialize the feauture. If not
            passed defaults to a list of one of the appropriate length
            
    names: a list of names corresponding to the params. If not passed, the list
            is populated with numbers.
    
    
    Important Methods:
    
    __call__(x,*params):
        returns self.func(x,*self.params), 
        if params is passed then self.params = list(*params)
    
    Iteration, getitem, and setitem treat the feature object like a dictionary
    with keys held in the names list and values held in the params list
    
    get_ methods allow for self reporting of parameters
    
    plot method allows for plotting
    """
         
    def __init__(self, function=None, ptype=None, params=None, names=None):
        self.ptype = ptype = str(ptype)
        
        #default construction
        if ptype in ptypes:
            self.func = funcs[ptype]
            self.names = ptypes[ptype]
            if params and len(params) == len(self):
                self.params = params
            else:
                self.params = [1]*len(self.names)
        
        #Overide with given values
        if function:
            if hasattr(function,"__call__"):
                self.func = function
        
            else:
                msg = 'Function object must have __call__ method'
                raise TypeError(msg)
                
        if names:
            self.names = names
                        
        if params:
            self.params = params
            
        self.verify(True)
            
    def verify(self,fix=False):
        """
        verify checks the feature's params and names lists and makes sure they
        are the proper length for the number of arguments taken by function.
        Raises an Index error is either list is wrong. If the fix kwarg is True,
        a error is not generated, instead the lists are populated with default
        values.
        """
    
        length = len(self)
        msg = ''
        
        for thing in ['params','names']:
            
            if hasattr(self,thing):
                attr = getattr(self,thing)
                
                if len(attr) != length:
                    if fix:
                    
                        if thing == 'params':
                            setattr(self,thing,[1]*length)
                            
                        if thing == 'names':
                            setattr(self,thing,[str(n) for n in range(length)])
                        
                    else:
                        msg += ' Length of {} is {}, should be {}'.format(thing,
                            len(attr),length)
                            
            else:
                if fix:
                    if thing == 'params':
                        setattr(self,thing,[1]*length)
                            
                    if thing == 'names':
                        setattr(self,thing,[str(n) for n in range(length)])
                        
                else:
                    msg += ' Error {} is undefined'.format(thing)         
                            
        else:
            if msg:
                    raise IndexError(msg)
    
    def __len__(self):
        return len(getargspec(self.func)[0])-1     
        
    def __repr__(self):
        rep = self.ptype
        for name in self:
            rep+=' '+str(name)+' '+str(self[name])
        return str(rep)
        
    def __iter__(self):
        return [name for name in self.names].__iter__()
        
    def __getitem__(self,pname):
        if pname in self.names:
            return self.params[self.names.index(pname)]
        else:
            raise IndexError
    
    def __setitem__(self,pname,val):
        if pname in self.names:
            self.params[self.names.index(pname)] = val
        else:
            raise IndexError
            
    def __call__(self, x, *params):
        """    
        returns a set of y-values that the feature would produce
        in a given x range. If passed a new set of parameters the parameters
        overwrite the old parameters. The y values are calculated based off the 
        new parameters.
        """
        
        if params:
            self.params = list(params)
            self.verify()
            
        else:
            params = self.params
            
        return self.func(x,*params)
         
    
    def get_results(self):
        results = {}
        for e in zip(self.names, self.params):
            results[e[0]] = e[1] 
        return results
        
    def get_named_results(self, name):
        results = self.get_results()
        for key in results.keys():
            results[name+' '+key] = results.pop(key)     
        return results
        
    def set_type(self, ptype=None):
        self.ptype = ptype
        
    def set_params(self, params=None):
        self.params = params
        
    def set_names(self, names=None):
        self.names = names
        
    def set_auto_names(self):    
        if self.ptype in ptypes:
            self.names = names[self.ptype]
        else:
            print 'Warning: Names Not Defined for {}'.format(self.ptype)
            
    def keys(self):
        return [name for name in self]
    
    def values(self):
        return [self[name] for name in self.names]
        
    def plot(self, x, ax=None, kwargs={}, offset = 0.):
        """
        The feature plotter function. 
        Labels each feature with its type and parameters unless another label is specified.
        May be passed a matplotlib axis and will plott the model on the axis
        Must be passed a valid series of x-values
        Full matplotlib kword arguments may be passed through the kward dictionary 
        """
        kill_label = False
        if 'label' not in kwargs.keys():

#            label = str(self.ptype) + ': '
#            for param in self.params:
#                label += str(round(param,3)) + ' '
                
            label = self.__repr__()
                
            kwargs['label'] = label
            kill_label = True
        
        if ax:
            ax.plot(x, self(x) + offset, **kwargs)
        else:
            plt.plot(x, self(x) + offset, **kwargs)

        if kill_label:
            del kwargs['label']
