# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:34:22 2013

@author: dgevans
"""
import numpy as np
from Spline import Spline

class ValueFunctionSpline:
    def __init__(self,X,y,k,sigma,beta):
        self.sigma = sigma
        self.beta = beta
        if sigma == 1.0:
            y = np.exp((1-beta)*y)
        else:
            y = ((1-beta)*(1.0-sigma)*y)**(1.0/(1.0-sigma))
        self.f = Spline(X,y,k)

    def fit(self,X,y,k):
        if self.sigma == 1.0:
            y = np.exp((1-self.beta)*y)
        else:
            y = ((1-self.beta)*(1.0-self.sigma)*y)**(1.0/(1.0-self.sigma))
        self.f.fit(X,y,k)

    def getCoeffs(self):
        return self.f.getCoeffs()

    def __call__(self,X,d = None):
        if d==None:
            if self.sigma == 1.0:
                return np.log(self.f(X,d))/(1-self.beta)
            else:
                return (self.f(X,d)**(1.0-self.sigma))/((1.0-self.sigma)*(1-self.beta))

        if sum(d)==1:
            return self.f(X)**(-self.sigma) * self.f(X,d)/(1-self.beta)
        raise Exception('Error: d must equal None or 1')



class DictWrap(object):
    """ Wrap an existing dict, or create a new one, and access with either dot
      notation or key lookup.

      The attribute _data is reserved and stores the underlying dictionary.
      When using the += operator with create=True, the empty nested dict is
      replaced with the operand, effectively creating a default dictionary
      of mixed types.

      args:
        d({}): Existing dict to wrap, an empty dict is created by default
        create(True): Create an empty, nested dict instead of raising a KeyError

      example:
        >>>dw = DictWrap({'pp':3})
        >>>dw.a.b += 2
        >>>dw.a.b += 2
        >>>dw.a['c'] += 'Hello'
        >>>dw.a['c'] += ' World'
        >>>dw.a.d
        >>>print dw._data
        {'a': {'c': 'Hello World', 'b': 4, 'd': {}}, 'pp': 3}

    """

    def __init__(self, d=None, create=True):
        if d is None:
            d = {}
        supr = super(DictWrap, self)
        supr.__setattr__('_data', d)
        supr.__setattr__('__create', create)

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            if not super(DictWrap, self).__getattribute__('__create'):
                raise
            value = {}
            self._data[name] = value

        if hasattr(value, 'items'):
            create = super(DictWrap, self).__getattribute__('__create')
            return DictWrap(value, create)
        return value

    def __setattr__(self, name, value):
        self._data[name] = value

    def __getitem__(self, key):
        try:
            value = self._data[key]
        except KeyError:
            if not super(DictWrap, self).__getattribute__('__create'):
                raise
            value = {}
            self._data[key] = value

        if hasattr(value, 'items'):
            create = super(DictWrap, self).__getattribute__('__create')
            return DictWrap(value, create)
        return value

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iadd__(self, other):
        if self._data:
            raise TypeError("A Nested dict will only be replaced if it's empty")
        else:
            return other