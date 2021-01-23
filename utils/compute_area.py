import numpy as np
from numpy import trapz
from scipy.integrate import simps

class ComputeArea():


    @staticmethod
    def auc_(x,y):
        '''
        # Compute the area using the composite trapezoidal rule.
        '''
        dx = x[1] - x[0]
        dydx = np.gradient(y, dx)
        #print(dydx)
        area = trapz(y, dx=dydx[0:10])
        return area

    @staticmethod
    def auc(x, y):
        '''
        # Compute the area using the composite trapezoidal rule.
        '''
        dx = x[1] - x[0]
        dydx = np.gradient(y, dx)
        # print(dydx)
        area = simps(y, dx=5)
        return area