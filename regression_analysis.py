import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import *
import scipy as sc
from scipy.stats import t
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mplc
import matplotlib as mpl


def detrend(time_series,order):
    time_series = time_series
    order       = order
    y            =    np.squeeze(np.asarray(time_series))
    if len(y.shape) !=1: 
        print("The input must be 1D array")
        
    else:
        nobs         =    y.shape[0]
        x            =    np.arange(1,nobs+1)
        z            =    np.polyfit(x,y,order)
        qhat         =    np.poly1d(z)
        slope        =    z[-2]
        intercept    =    z[-1]

        if order>1:
            print("warning : This slope is from nonlinear fit")

        detrended   =     y - qhat(x) 
           
    
    return detrended,slope,intercept
    

def linear_regress(xx,yy,p):
    yy                    =    np.squeeze(np.asarray(yy))
    xx                    =    np.squeeze(np.asarray(xx))

    if len(yy)!=len(xx):
        print("Length of the arrays mismatch !")
    else:
        N                     =         len(xx)
        x,s1,i1               =         detrend(xx,order=1)
        y,s2,i2               =         detrend(yy,order=1)
        #print(x,y)
        corr                  =         np.corrcoef(x,y)[0,1]
        slope                 =         np.polyfit(x,y,1)[0]
        intercept             =         np.polyfit(x,y,1)[1]
        qhat                  =         slope*x+intercept
        error                 =         (y-qhat)**2
        var_residuals         =         (1/(N-2))*(np.sum(error))
        x_er                  =         np.sum((x-np.mean(x))**2)
        s                     =         var_residuals/x_er
        standard_error        =         (s)**0.5
        t_score               =         np.absolute(slope)/standard_error
        # Student t-distribution Percent Point Function
        # define probability
        #p = 0.025
        df                    =         N-2
        # retrieve value <= probability
        t_critical            =         t.ppf(1-p/2, df)
        # confirm with cdf
        p1                    =        1.- t.cdf(t_score, df)

    return slope,intercept,p1,corr 


def write_to_netcdf(data_map,output_filename,ex_filename='',varname_ex=''):
    ## for 3D file structure
    d         =   data_map
    ds1       =   xr.open_dataset(ex_filename)
    times     =   ds1.time
    lon       =   ds1.lon
    lat       =   ds1.lat
    dk2       =   ds1

    t         =   xr.DataArray(d,coords=[('time', times),('lat', lat),('lon', lon)])
    dk2[varname_ex]=t
    print (dk2.coords)
    print ('finished saving')
    dk2.to_netcdf(output_filename)

    return  print("Thank you")


class reg_plot(): 
    
    """Plotting the regression map with significance """
    """Plotting the correlation map with significance"""
   
    
    def __init__(self,time_series=[],data_name='',varname_data='',p=0):
        self.time_series  = time_series 
        self.data_name    = data_name
        self.varname_data = varname_data
        self.p            = p
    
    def explain_to(self):
        print("Hello, users. These are inputs:")
        print("Your time series is {}.".format(self.time_series))
        print("Your data filename is {}.".format(self.data_name))
        print("Your variable name is {}.".format(self.varname_data))
        print("confidence value is {}.".format(self.p))
        

    
    def regression_map_making(self):

        data_f     =      xr.open_dataset(self.data_name)
        data       =      data_f[self.varname_data].values
        shape      =      [1,data.shape[1],data.shape[2]]
        regress_map=      np.zeros((shape))
        cor_map    =      np.zeros((shape))
        significant_map=  np.zeros((shape))

        for i in range(data.shape[2]):
            for j in range(data.shape[1]): 
                temp              = data[:,j,i]
                if np.all(np.isnan(temp)):
                    regress_map[0,j,i]= np.nan
                    cor_map[0,j,i]    = np.nan
                else:
                    slope,intercept,p1,corr=linear_regress(self.time_series,temp,self.p)
                    regress_map[0,j,i]= slope
                    cor_map[0,j,i]    = corr
                    
                    if p1<self.p/2:
                        significant_map[0,j,i] = 1
                    else:
                        significant_map[0,j,i] = 0
        
        return regress_map,cor_map,significant_map    

    
    def draw_regression(self,vmin,vmax,inc,titlestr,cmap='RdBu',hatch='/'):
        regress_map,cor_map,significant_map= self.regression_map_making()
        ds1           =     xr.open_dataset(self.data_name)
        lon           =     ds1.lon
        lat           =     ds1.lat

        # m = Basemap(projection='ortho',lat_0=0,lon_0=-180,resolution='l')
        #m = Basemap(projection='moll',lon_0=0,lat_0=0,resolution='l')  

        m              = Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(),urcrnrlon =lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(),resolution='c')
        lon2, lat2     =    np.meshgrid(lon,lat)
        x, y           =    m(lon2, lat2)
        fig            =    plt.figure(figsize=(10,7))

        #fig = plt.figure()
        #m.fillcontinents(color='gray',lake_color='gray')

        m.drawcoastlines()
        m.drawparallels(np.arange(-80.,81.,20.),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180.,181.,60.),labels=[0,0,0,1])

        #m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
        #m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        # m.drawmapboundary(fill_color='white')
        norm      =    mpl.colors.Normalize(vmin,vmax)
        v         =    np.arange(vmin,vmax+inc,inc)
        cs        =    m.contourf(x,y,regress_map[0,:,:],v,norm=norm,extend='both',cmap=plt.cm.get_cmap(cmap))
        #levels=[0,1]
        zm = np.ma.masked_equal(significant_map[0,:,:], 0)
        m.contourf(x,y,zm, hatches=hatch,alpha=0.)
        cbar0     = plt.colorbar(cs,orientation='horizontal',fraction=0.05)
        plt.title(titlestr)

        return print("myferret")
    
    
    def draw_correlation(self,significant_value,vmin,vmax,inc,titlestr,cmap='RdBu',hatch='/'):
        regress_map,cor_map,significant_map=self.regression_map_making()
        ds1=xr.open_dataset(self.data_name)
        lon=ds1.lon
        lat=ds1.lat
        # m = Basemap(projection='ortho',lat_0=0,lon_0=-180,resolution='l')
        #m = Basemap(projection='moll',lon_0=0,lat_0=0,resolution='l')  
        m = Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(),         urcrnrlon =      lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(), resolution='c')
        lon2, lat2 = np.meshgrid(lon,lat)
        x, y       = m(lon2, lat2)
        fig        = plt.figure(figsize=(10,7))
        #fig = plt.figure()
        #m.fillcontinents(color='gray',lake_color='gray')
        m.drawcoastlines()
        m.drawparallels(np.arange(-80.,81.,20.),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180.,181.,60.),labels=[0,0,0,1])
        #m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
        #m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        # m.drawmapboundary(fill_color='white')
        norm = mpl.colors.Normalize(vmin,vmax)
        v=np.arange(vmin,vmax+inc,inc)
        cs = m.contourf(x,y,cor_map[0,:,:],v,norm=norm,extend='both',cmap=plt.cm.get_cmap(cmap))
        #levels=[0,1]
        #m.contourf(x,y,sig[0,:,:], 2, hatches=["", "/"],alpha=0)
        m.contour(x, y, cor_map[0,:,:], levels=[-1*significant_value,significant_value], linewidths=0.5, colors='black', antialiased=True)
        cbar0 = plt.colorbar(cs,orientation='horizontal',fraction=0.05)
        plt.title(titlestr) 
        plt.show()
        return print("myferret")
        
        
    

