import hashlib
import os

from numpy import trapz
import numpy as np
import matplotlib.pyplot as plt

#import pandas as pd 


import sys
filename = sys.argv[1]


def Generate_ID():
    id = "W20075826"
    for part in ['a','b','c']:
        code = id + '-' + part
        seed = int(hashlib.sha512(str(code).encode()).hexdigest(),16) % 2**32
        filename = "Data/p-%010d.txt" % seed  
        print (id, part, filename, os.path.isfile(filename))
        
        

def display_plot(data, name):
    

    plt.figure(figsize=(16,10))
    plt.xlabel("omega")
    plt.ylabel("f")
    plt.plot(data[:,0], data[:,1])
    plt.title("File name: %s" % name)    
#    plt.savefig("%s.pdf" % name, bbox_inches="tight")
    plt.show()
    

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
 

       
def main(name_Of_File):
    filename = name_Of_File
    data = np.loadtxt(filename)

    #print(data.dtype, data.shape)
    assert data.shape[0] == 1000, "Expected 1000 data points"
    assert data.shape[1] == 2, "Expected 2 columns of data"
    
    
    display_plot(data, filename)
    
#    x = data[:,0] #omega
#    y = data[:,1] #f
    x_raw=data[:,0]
    y_raw=data[:,1]
 
    y = movingaverage(y_raw,3)
    x = x_raw[1:-1]

  
    trend_coeff = np.polyfit(x, y,2)
    trend_poly=np.poly1d(trend_coeff)
    trend = trend_poly(x)
   


    y_p = np.abs(y - trend) 

    T1 = 0.2
    nonpeaks = y_p < T1
    filter_pointsx = x[nonpeaks]
    filter_pointsy = y[nonpeaks]
    trend_coeff2 = np.polyfit(filter_pointsx,filter_pointsy,2)
    trend_poly2=np.poly1d(trend_coeff2)
    trend2 = trend_poly2(filter_pointsx)
#    plt.figure(figsize=(16,10))
#    plt.plot(data[:,0], data[:,1], label ='Original Raw Data')
#    plt.plot(data[:,0], trend, 'g', label = 'Trend - with Original Peaks') #green - original trend
#    plt.plot(filter_pointsx,filter_pointsy,label = 'Raw Data - No peaks') 
#    plt.plot(filter_pointsx, trend2,'k',label='Trend - No Peaks') #red - new trend with peaks gone
#    plt.legend()
  
    c1 = np.round(trend_coeff2[0],3)
    c2 = np.round(trend_coeff2[1],3)
    c3 = np.round(trend_coeff2[2],3)


    trend_points = []
    for i in range(len(x)):
        trend_points.append(c1*x[i]**2 + c2*x[i] + c3)
        i = i+1 

    y_new = abs(y-trend_points)
    Tnew = 0.1
    peak = y_new > Tnew
    d_peaks = np.diff(np.array(peak,dtype=int)) #diff leaves d_peaks with one less value in set, if peak has 1000 values, d_peaks will have 999
    
    lefts = d_peaks == 1
    sum_of_lefts = np.sum(lefts)
    rights = d_peaks == -1
    sum_of_rights = np.sum(rights)
    assert np.sum(lefts)==np.sum(rights)
    peaks_start = x[:-1][lefts]
    peaks_end = x[:-1][rights]

    plt.figure(figsize=(16,10))
    plt.title("Trendline with Time Averaged Data" )   
    plt.plot(x,y,"b", label = "Original")
    plt.plot(x,trend, "r", label = "Trend 1")
    plt.plot(x,trend_points, "y", label ="Final Trend")
    plt.legend()
    plt.show()
    #plt.figure(figsize=(16,10))
    #plt.title("Data - Trendline ")   
    #plt.plot(x, y_p, "k", label = "Trend 1 applied")
    #plt.plot(x,y_new, "b", label = "Final Trend applied")
    #plt.axhline(y=0.1, color='r', linestyle='-',label = "Threshold1")
    #plt.axhline(y=0.2, color='r', linestyle='-',label = "Threshold2")
    #plt.legend()
    #plt.show()

    
    peakCount = np.sum(lefts)
    print("The Trend: ", str(c1) + 'x^2 +' + str(c2) + 'x +'+ str(c3))
    print("Number of peaks:", sum_of_lefts)
    Center = []
    Width = []
    Height = []
    Area = []
    for p in range(peakCount):
        start = peaks_start[p]
        end = peaks_end[p]
        c = (end + start)/2
        w = (end - start)
        h = np.max(y_new[(start<=x) & (x<end)])
        a = trapz(y_new[(start<=x) & (x<end)],x[(start<=x) & (x<end)])
        print("Peak %d: c=%.4f w=%.4f h =%.4f a=%.4f" % (p,c,w,h,a))
        Center.append(c)
        Width.append(w)
        Height.append(h)
        Area.append(a)
        
    
   
    np.savetxt('data.csv', (Center, Width, Height, Area), delimiter=',')
    
   


main(filename)