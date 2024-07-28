#! /usr/bin/python3

"""
Author: Burak Ulas -  github.com/burakulas
2024, Konkoly Observatory, COMU
"""

import glob
import os
import shutil
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os

os.mkdir('pngs_txts')  ## This is the output directory

fre2 = 0
amp2i = 0
fi2 = 0

fi1 = 0
fillist = (glob.glob("*.dat"))  ## from the output of 'det_synth_lc.sh' or 'sdet_synth_lc.sh' 
sayi = 1
    
def art_pul(cyc1, cyc2, cyc3, ain, afi, ainc, dpis):
    for file in fillist:
        valu = file[5:21].replace('d', 'e')
        #per = float(file[5:21])
        per = float(valu)
        num = 1
        fa = 1 / (0.3 * per / cyc1)
        fb = 1 / (0.3 * per / cyc2)
        fc = 1 / (0.3 * per / cyc3)
        for fre1 in (fa, fb, fc):
            for amp1i in np.arange(ain, afi, ainc): 
                #print("no: ", sayi," Per: ",per," fr: ", fre1, "c/d", " A: ", amp1i, "mag.")    
                dlc = pd.read_csv(file, header=None, delimiter=r"\s+")
                dlc.columns = ['ph','lmpre']
                
                xval = dlc['ph'].to_numpy()              
                datnr = dlc[(dlc['ph'] >= 0.7) & (dlc['ph'] <= 0.8)]
                yval = datnr['lmpre'].to_numpy()
                stdnr = datnr['lmpre'].std()
                xvalorj = dlc['ph'].to_numpy()
                yvalorj = dlc['lmpre'].to_numpy()
                noise = np.random.normal(0, 1 * stdnr, len(xvalorj)) #  
                yval_noi = yvalorj + noise
                dfnoi = pd.DataFrame(yval_noi, columns=['FLUX'])
                dlc['lm'] = dfnoi['FLUX']
                dlc2 = dlc.sort_values(by=['ph'])
                amp1 = (10 ** (-0.4 * amp1i)) - 1
                amp2 = (10 ** (-0.4 * amp2i)) - 1
                #amp2 = 10 ** (-0.4 * amp2i)
                dlc2['tim'] = dlc2.apply(lambda row: row.ph * per, axis=1)
                dlc2['fac'] = (amp1 * np.sin(2 * math.pi * (fre1 * dlc2['tim'] + fi1)) + amp2 * np.sin(2 * math.pi * (fre2 * dlc2['tim'] + fi2))).where(dlc2['ph'].between(0.1, 0.4), 0) + (amp1 * np.sin(2 * math.pi * (fre1 *dlc2['tim'] + fi1)) + amp2 * np.sin(2 * math.pi * (fre2 * dlc2['tim'] + fi2))).where(dlc2['ph'].between(0.6, 0.9), 0) + (amp1 * np.sin(2 * math.pi * (fre1 *dlc2['tim'] + fi1)) + amp2 * np.sin(2 * math.pi * (fre2 * dlc2['tim'] + fi2))).where(dlc2['ph'].between(1.1, 1.25), 0)  ### addition factors at maximum phases
                dlc2['pul'] = dlc2['lm'] + dlc2['fac']
                dlc3 = dlc2[(dlc2['ph'] >= 0.60) & (dlc2['ph'] <= 0.90)]
                dlc4 = dlc2[(dlc2['ph'] >= 0.95) & (dlc2['ph'] <= 1.05)]
                dlc6 = dlc2[(dlc2['ph'] >= 0.7) & (dlc2['ph'] <= 0.8)]

                stdnr0 = dlc3['lm'].std()
                stdnr3 = dlc3['pul'].std()
                stdnr6 = dlc6['pul'].std()

                minfac = dlc6['fac'].min()
                maxfac = dlc6['fac'].max()

                min3 = dlc3['pul'].min()
                max3 = dlc3['pul'].max()

                min2 = dlc2['lm'].min()
                max2 = dlc2['pul'].max()

                min4 = dlc4['lm'].min()
                max4 = dlc4['lm'].max()
                
                min6 = dlc6['pul'].min()
                max6 = dlc6['pul'].max() 

                xmin = 0.5 
                xmax = 1.25

                ymin = min2 - ((max3 - min3) / 2)
                ymax = max2 + ((max3 - min3) / 2)
                
                xaxis = xmax - xmin # length of xaxis
                yaxis = ymax - ymin # length of yaxis                

                if (max6 - min6) < (max4 - min4): # pulsation amplitude always smaller than min
                
                    ## Figure plot
                    dlc5 =  dlc2[(dlc2['ph'] >= 0.50) & (dlc2['ph'] <= 1.25)]   
                    sizep = 5 * 72 / dpis # pointsize fixed to 5 pixel
                    ax = plt.gca()
                    ax.scatter(dlc5['ph'], dlc5['pul'], marker=".", lw=0, c='Black', s=sizep)
                    ax.set_axis_off()
                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                    plt.margins(0,0)
                    plt.ylim(ymin, ymax)
                    plt.gcf().set_size_inches(1, 1)
                    #dpis = 240
                    plt.savefig('pngs_txts/{}_{}_{}_{}_lc.png'.format(per,fre1,amp1i,file),dpi=dpis)
                    plt.close()
                    img = Image.open('pngs_txts/{}_{}_{}_{}_lc.png'.format(per,fre1,amp1i,file))
                    os.remove('pngs_txts/{}_{}_{}_{}_lc.png'.format(per,fre1,amp1i,file)) 
                    img = img.convert("L")  ## convert to grayscale
                    img.save('pngs_txts/{}_{}_{}_{}_lc.png'.format(per,fre1,amp1i,file))
                                    
                    ## Pulsation annotation
                    classp = 0
                    x_centerp = (0.75 - xmin) / xaxis # xmin is starting point
                    widthp = 0.30  / xaxis #
                    y_center = ((max3 + min3) / 2)
                    y_centerp = (ymax - y_center) / yaxis
                    heightp = 1.5 * (max3 - min3) / yaxis
         

                    ## Minimum annotation
                    classm = 1
                    x_centerm = (1.0 - xmin) / xaxis 
                    widthm = 0.15 / xaxis #
                    y_center2 = ((max2 + min2) / 2)
                    y_centerm = (ymax - y_center2) / yaxis
                    heightm = (max2 - (min2 - ((max3 - min3) / 2))) / yaxis
                  
                    f = open(f"pngs_txts/{per}_{fre1}_{amp1i}_{file}_lc.txt", 'w')
                    text = f"{classp} {x_centerp} {y_centerp} {widthp} {heightp}\n{classm} {x_centerm} {y_centerm} {widthm} {heightm}"
                    lines = text.split('\n')
                    print('\n'.join(lines), file=f)
                    num += 1



art_pul(10, 7, 5, 20e-3, 50e-3, 10e-3, 240) # 10, 7, 5 : number of cycles in 0.3 phase interval, A_initial, A_final, A_increment, dpi 
