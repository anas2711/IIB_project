# functions for the main program

#import image from the computer
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imageio
from findpeaks import findpeaks
import scipy as sp
from scipy.stats import norm
#new functions for this notebook
from sklearn.neighbors import KernelDensity
import cv2
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, peak_widths

from SpecularAnalysis import *


# vol properties by number of pixels 

def print_color_from_rgb(r,g,b,size=30):
    final_color=np.ones((size,size,3))
    final_color[:,:,0]=  r
    final_color[:,:,1]=  g
    final_color[:,:,2]=  b 
    #stack 
    final_color = cv.convertScaleAbs(final_color)
    return final_color

def print_color_from_hsv(h,s,v,size=30):
    final_color=np.ones((size,size,3))
    final_color[:,:,0]=  h
    final_color[:,:,1]=  s
    final_color[:,:,2]=  v 
    final_color = cv.convertScaleAbs(final_color)
    final_color_RGB = cv.cvtColor(final_color, cv.COLOR_HSV2RGB)
    return final_color_RGB
    
def show_vp(volume_properties, top=0, show=0):
    if top==0:
        top=len(volume_properties)
    volume_properties.sort(key=lambda x: x[0], reverse=True)
    plt.figure()
    for i in range(0,top):
        #divide by 5 and round up
        plt.subplot(len(volume_properties)//5 +1, 5, i+1)
        color_image = print_color_from_rgb(volume_properties[i][2],volume_properties[i][3],volume_properties[i][4])
        plt.imshow(color_image)
        # plt.title(str(volume_properties[i][0])+ ',\n rgb: '+ str(volume_properties[i][2])+','+str(volume_properties[i][3])+','+str(volume_properties[i][4]))
        #dont show axis
        plt.axis('off')
        #make titles small
        plt.rcParams.update({'font.size': 6})
        plt.subplots_adjust(right=0.8)

def show_mp(material_properties, top=0, show=0):
    if top==0:
        top=len(material_properties)
    material_properties.sort(key=lambda x: x[0], reverse=True)
    plt.figure()
    for i in range(0,top):
        #divide by 5 and round up
        plt.subplot(len(material_properties)//5 +1, 5, i+1)
        r,g,b= material_properties[i][2]
        color_image = print_color_from_rgb(r,g,b)
        plt.imshow(color_image)
        
        plt.axis('off')
        #make titles small
        plt.rcParams.update({'font.size': 6})
        plt.subplots_adjust(right=0.8)
       

def display_hue(hue):

            # show the color with different V and S
            color_prominence= np.ones((255,255,3))

            color_prominence[:,:,0]= hue
            for i in range(0,255):
                color_prominence[:,i,1]= color_prominence[:,i,1]* i #S
                color_prominence[i,:,2]= color_prominence[i,:,2]* (255-i) # V

            color_prominence = cv.convertScaleAbs(color_prominence)
            color_prominence_RGB = cv.cvtColor(color_prominence, cv.COLOR_HSV2RGB)
            plt.figure()
            plt.imshow(color_prominence_RGB)
            plt.xlabel('Increasing Saturation --> ')
            plt.ylabel('Increasing Value -->')
            #set ticks to zero
            plt.xticks([])
            plt.yticks([])
            plt.title('The extracted hue of the object')
# current functions

def display_in_green(masked_rgb_img):
    masked_image_display= np.copy(masked_rgb_img)
    for i in range(0, masked_rgb_img.shape[0]):
        for j in range(0, masked_rgb_img.shape[1]):
            if masked_rgb_img[i,j,2] == 0: #value is zero
                masked_image_display[i,j,:]= [0,255,0]
    return masked_image_display

def count_pixels_of_image(saved_masked_img):
    pixel_total_masked = 0
    for i in range(0, saved_masked_img.shape[0]):
        for j in range(0, saved_masked_img.shape[1]):
            if saved_masked_img[i,j,2] != 0: #value is not zero
                pixel_total_masked += 1
    return pixel_total_masked

def count_pixels_of_mask(mask):
    pixel_total_masked = 0
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            if mask[i,j] != 0: #value is not zero
                pixel_total_masked += 1
    return pixel_total_masked

def thresholding(threshold, historgram):
    pulse_start_end= []
    hist_pulse_area= []
    hist_tr= historgram > threshold
    hist_tr = hist_tr.astype(int)

    hist_total_pulse_area= np.sum(hist_tr)
   # add tolerance to the thresholding
    max_tol= 1
    tol1=0
    tol2=0
    if hist_tr[0]==1:
        pulse_start_end.append(0)
    for i in range(1, len(hist_tr)):
        if hist_tr[i] == 1 and hist_tr[i-1] == 0:
            tol1=tol1 + 1
            if tol1==max_tol:
                pulse_start_end.append(i)
                tol1=0
        if hist_tr[i] == 0 and hist_tr[i-1] == 1:
            tol2=tol2 + 1
            if tol2==max_tol:
                pulse_start_end.append(i)
                tol2=0

    if hist_tr[-1]==1:
        pulse_start_end.append(len(hist_tr))

    for i in range(0, len(pulse_start_end)//2 ):
        hist_pulse_area.append(np.sum(hist_tr[pulse_start_end[2*i]:pulse_start_end[2*i+1]]))
        hist_tr[pulse_start_end[2*i]:pulse_start_end[2*i +1]] = hist_pulse_area[i]
    return hist_tr, hist_pulse_area, pulse_start_end

def make_new_image(rgb_img, mask):
    red = rgb_img[:,:,0]*mask
    green = rgb_img[:,:,1]*mask
    blue = rgb_img[:,:,2]*mask
    masked_img = np.dstack((red,green,blue))
    return masked_img

def make_material_image(masked_img,hue, saturation,value, material_colors_img):

    # make a new image with the material color
    # for each pixel in the masked image that is not zero, set the hue, saturation, and value
    # of the pixel in the material color image to the hue, saturation, and value of the pixel
    shape= masked_img.shape
    if shape[-1]==3:
        for i in range(0, masked_img.shape[0]):
            for j in range(0, masked_img.shape[1]):
                if masked_img[i][j][0] != 0: # if the pixel is not black
                    material_colors_img[i][j][0] = hue
                    material_colors_img[i][j][1] = saturation
                    material_colors_img[i][j][2] = value
    if shape[-1]==2:
        for i in range(0, masked_img.shape[0]):
            for j in range(0, masked_img.shape[1]):
                if masked_img[i][j] > 0: # if the pixel is not black
                    material_colors_img[i][j][0] = hue
                    material_colors_img[i][j][1] = saturation
                    material_colors_img[i][j][2] = value
    return material_colors_img

def kde(x,total_pixels_img,h, sampling=np.linspace(-10,10,100)):
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(x)
    log_prob= kde.score_samples(x)
    
    return (sampling,log_prob)


def n_pixels_of_peak(histogram, peaks):
    #find bases where gradient is zero
    #calculate the number of pixels of the peak on each side of the peak until the bases 
    histogram_gradient= np.gradient(histogram)
    

    n_left_of_peak = np.zeros(len(peaks))
    n_right_of_peak = np.zeros(len(peaks))
    left_base = np.zeros(len(peaks))
    right_base = np.zeros(len(peaks))

    for b in range(len(peaks)):
        # find left base
        #previous_value = histogram[peaks[int]] 
        location= int(peaks[b])

        n_left_of_peak[b]= (histogram[location])/2
        if location == 0:
              n_left_of_peak[b]= histogram[location]
        i  = location - 1
        while histogram_gradient[i]  > 0 and i > 0:
            n_left_of_peak[b] += histogram[i]
            #previous_value = histogram[i]
            i=i-1
        left_base[b] = i

        # find right base
        #previous_value = histogram[peaks[int]]
        n_right_of_peak[b]= histogram[location]/2
        if location == len(histogram)-1:
           n_right_of_peak[b]= histogram[location]
        i  = location + 1
        while histogram_gradient[i]  <0  and i < (len(histogram)-2):
            n_right_of_peak[b] += histogram[i]
            #previous_value = histogram[i]
            i=i+1
        right_base[b] = i
   
    return n_left_of_peak, n_right_of_peak, left_base, right_base

def new_bases(histogram, peaks, n_left_of_peak, n_right_of_peak, left_base, right_base):

    #calculate the new bases for the peaks

    new_left_bases = np.zeros(len(peaks))
    new_right_bases = np.zeros(len(peaks))

    for c in range(len(peaks)-1):
        counter = 0
        location= int(peaks[c])
    
        while n_left_of_peak[c]>0 and (location-counter)> left_base[c]:
            n_left_of_peak[c] -= histogram[(location-counter)]
            counter += 1

        new_left_bases[c] = location - counter

        counter=0
        while n_right_of_peak[c]>0 and ( location +counter)<right_base[c]:
            n_right_of_peak[c] -= histogram[location +counter]
            counter += 1

        new_right_bases[c] = location + counter

    return new_left_bases, new_right_bases

def remove_small_peaks(histogram,peaks,threshold):
    #remove peaks that are smaller than threshold
    new_peaks = np.zeros(len(peaks))
    counter=0
    for i in range(len(peaks)):
        if histogram[peaks[i]] > threshold:
            new_peaks[counter] = int(peaks[i])
            counter += 1
    new_peaks = new_peaks[:counter]
    new_peaks = new_peaks.astype(int)
    return new_peaks

def calc_hue_kde(hue_values, total_pixels_img, show= False):
    # Plot the histogram log scale
    plt.figure()
    histogram= plt.hist(hue_values, bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.yscale('log')
    plt.title('Histogram of Hues')
    plt.xlabel('Hue Value')
    plt.ylabel('Log of Frequency')

    # Calculate the KDE (Kernel Density Estimation) of hues
    kde = gaussian_kde(hue_values, bw_method=0.0325)

    # Plot the KDE
    x_integers = np.arange(0, 256)
    peaks,fp_scipy = find_peaks(kde(x_integers), distance=7)
    for peakindex in range(len(peaks)):
        peaks[peakindex]= peaks[peakindex]
    peaks= remove_small_peaks(histogram[0],peaks, total_pixels_img*0.0001)
    kde_histogram= kde(x_integers)*total_pixels_img
   
    n_left_of_peak, n_right_of_peak, l_bases, r_bases = n_pixels_of_peak( kde_histogram, peaks) 
    new_left_bases, new_right_bases= new_bases(histogram[1], peaks, n_left_of_peak, n_right_of_peak, l_bases, r_bases)
    
    if 1:
        # Plot the histogram
        '''plt.figure()   
        plt.plot(x_integers, kde_histogram, color='red', alpha=0.7)
        plt.scatter(x_integers[peaks], kde_histogram[peaks], color='blue')
        plt.fill_between(x_integers, 0, kde_histogram, color='red', alpha=0.1)
        plt.scatter( l_bases, kde_histogram[l_bases.astype(int)], color='green', marker='>')
        plt.scatter( r_bases, kde_histogram[r_bases.astype(int)], color='green', marker='<')
        plt.title('KDE of Hues')
        plt.xlabel('Hue Value')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.show()'''

        plt.figure()   
        plt.plot(x_integers, kde_histogram, color='red', alpha=0.7)
        plt.scatter(x_integers[peaks], kde_histogram[peaks], color='blue', label='Peaks')
        plt.fill_between(x_integers, 0, kde_histogram, color='red', alpha=0.1)
        plt.scatter( new_left_bases, np.zeros(len(new_left_bases)), color='red', marker='>', label='Segmentation start')
        plt.scatter( new_right_bases, np.zeros(len(new_left_bases)), color='red', marker='<', label='Segmentation end')
        #plt.title('KDE of Hues')
        plt.xlabel('Hue')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

        #show histogram
        plt.figure()
        plt.hist(hue_values, bins=256, range=[0, 256], color='blue', alpha=0.7)
        plt.scatter(peaks, np.zeros(len(peaks)), color='red', label='Peaks')
        plt.scatter(new_left_bases, np.zeros(len(new_left_bases)), color='green', marker='>', label='Segmentation start')
        plt.scatter(new_right_bases, np.zeros(len(new_right_bases)), color='green', marker='<', label='Segmentation end')
        plt.yscale('log')
        #plt.title('Histogram of Hues')
        plt.xlabel('Hue')
        plt.ylabel('Log of Frequency')
        plt.legend()
        plt.show()



    #sort based on number of pixels in each peak in decending order
    peaks_sorted= peaks[(-kde_histogram[peaks]).argsort( )]
    new_left_bases_sorted= new_left_bases[(-kde_histogram[peaks]).argsort( )]
    new_right_bases_sorted= new_right_bases[(-kde_histogram[peaks]).argsort( )]

    return peaks_sorted, new_left_bases_sorted, new_right_bases_sorted

def calc_sat_kde(sat_values, total_pixels_img, show=False):
    plt.figure()  

    # Calculate the KDE (Kernel Density Estimation) of hues
    kde = gaussian_kde(sat_values, bw_method=0.05)

    # Plot the KDE
    x_integers = np.arange(0, 256)
    kde_histogram=  kde(x_integers)*total_pixels_img
    #find minima from the kde
    local_minima = np.where((kde_histogram[1:-1] < kde_histogram[0:-2]) * (kde_histogram[1:-1] < kde_histogram[2:]))[0] + 1
    # add zero and 255 to the local minima if they are not already there
    #if there are no local minima, add zero and 255
    if len(local_minima)==0:
        local_minima= np.array([255])
    if local_minima[-1] != 255:
        local_minima= np.append(local_minima, 255)
    #add zero to beginning of the array if there is no zero already

    if local_minima[0] != 0:
        local_minima= np.insert(local_minima, 0, 0) #exclude zero?
    #number of pixels between each minima
    saturation_range= np.zeros((len(local_minima)-1,3))
    for i in range(0, len(local_minima)-1):
        pixels= np.sum(kde_histogram[local_minima[i]:local_minima[i+1]])
        saturation_range[i]= [pixels, local_minima[i], local_minima[i+1]]
    #sort by number of pixels in decending order
        
    saturation_range= saturation_range[(-saturation_range[:,0]).argsort( )]
    #total pixels
    total_pixels= np.sum(kde_histogram)
    #how many arrays make up 80% of the pixels
    counter=0
    pixels=0
    while pixels< total_pixels*0.90:
        pixels= pixels+ saturation_range[counter,0]
        counter= counter+1
        if counter==len(saturation_range):
            break
    
    #take the range of the 80% of the pixels
    saturation_range2= saturation_range[0:counter]
    #print how many ranges make up 80% of the pixels
    if show:
        print('number of ranges that make up 90% of the pixels', counter)
        print('out of', len(saturation_range), 'ranges in total')

    s_pulse_start_end= []
    #new segmentation
    for i in range(0, len(saturation_range2)):
        s_pulse_start_end.append(saturation_range2[i,1])
        s_pulse_start_end.append(saturation_range2[i,2])



    #thresholding 
    s_zeros= np.zeros(len(s_pulse_start_end))
    if show:

        plt.figure()   
        plt.plot(x_integers, kde_histogram, color='blue', alpha=0.7)
        plt.scatter( local_minima, kde_histogram[local_minima], color='green', marker='|', label='Local minima')
        plt.scatter(saturation_range2[:,1], kde_histogram[saturation_range2[:,1].astype(int)], color='red', marker='>', label='Segmentation start')
        plt.scatter(saturation_range2[:,2], kde_histogram[saturation_range2[:,2].astype(int)], color='red', marker='<', label='Segmentation end')
        #plt.title('KDE of Saturation')
        plt.xlabel('Saturation')
        plt.ylabel('Number of pixels')
        plt.legend()
        plt.show()

    return kde_histogram, s_pulse_start_end

def calc_val_kde(val_values, total_pixels_img, percentage, show=False):
    # remove value is zero 
    val_values= val_values[val_values>0]
    v_end=[]
    segmentation=[]
    # Calculate the KDE (Kernel Density Estimation) of hues
    if len(val_values)==0:
        if show:
            print('no value detected:set to 0')
        return 0, 0
    kde = gaussian_kde(val_values)

    # Plot the KDE
    x_integers = np.arange(0, 256)
    kde_histogram=  kde(x_integers)*total_pixels_img
    local_minima = np.where((kde_histogram[1:-1] < kde_histogram[0:-2]) * (kde_histogram[1:-1] < kde_histogram[2:]))[0] + 1

    if len(local_minima)==0:
        local_minima= np.array([255])
    if local_minima[-1] != 255:
        local_minima= np.append(local_minima, 255)

    if local_minima[0] != 0:
        local_minima= np.insert(local_minima, 0, 0) #exclude zero?
    
    #segmetnation based on the percentage of pixels
    value_range= np.zeros((len(local_minima)-1,3))
    for i in range(0, len(local_minima)-1):
        pixels= np.sum(kde_histogram[local_minima[i]:local_minima[i+1]])
        value_range[i]= [pixels, local_minima[i], local_minima[i+1]]
    #sort by number of pixels in decending order
    value_range= value_range[(-value_range[:,0]).argsort( )]
    #total pixels
    total_pixels= np.sum(kde_histogram)
    #how many arrays make up 70% of the pixels
    counter=0
    pixels=0
    while pixels< total_pixels*percentage:
        pixels= pixels+ value_range[counter,0]
        counter= counter+1
        if counter==len(value_range):
            break
    #take the range of the 70% of the pixels
    value_range2= value_range[0:counter]
    #sort by start value
    value_range2= value_range2[(value_range2[:,1]).argsort( )]
    #if end of one is the start of another merge them
    i=0
    while i <  len(value_range2)-1:
        if value_range2[i,2]== value_range2[i+1,1]:
            value_range2[i,2]= value_range2[i+1,2]
            value_range2[i,0]= value_range2[i,0]+ value_range2[i+1,0] #total pixels
            value_range2= np.delete(value_range2, i+1, axis=0)
        else:
            i=i+1
            

    #print how many ranges make up 70% of the pixels
    if show:
        print('number of ranges that make up 70% of the pixels', counter)
        print('out of', len(value_range), 'ranges in total')
   
    #for each pulse between the start and end, find value at which 90% of the pixels are below
    #cumulative distribution function
    cdf = np.cumsum(kde_histogram)
    for range_sample in range(0, len(value_range2)):
        end=int( value_range2[range_sample,2])
        n_pixels= 0.1*value_range2[range_sample,0]
        #go bckwards from end until n_pixels is reached
        segmentation.append([value_range2[range_sample,1], end])
        pixel_counter=0
        for i in range(end,0,-1):
            pixel_counter= pixel_counter+ kde_histogram[i]
            if n_pixels <= pixel_counter:
                v_end.append(i)
                break
    #make segmentation numpy array
    segmentation= np.array(segmentation)
    return kde_histogram, v_end, segmentation



def remove_black_zeros(huearray,satarray, valuearray):
    newhuearray= []
    newsatarray= []
    newvaluearray= []
    for i in range(len(huearray)):
        if valuearray[i] != 0:
            newhuearray.append(huearray[i])
            newvaluearray.append(valuearray[i])
            newsatarray.append(satarray[i])
    return newhuearray, newsatarray, newvaluearray

def hsv_analysis_new(rgb_img, Specular, material_properties, show_graphs= False):

    material_colors_img = np.zeros((rgb_img.shape[0],rgb_img.shape[1],3))
    #add black and white from previous step
    if len(material_properties) != 0:
        for material in material_properties:
            #rgb to hsv
            material_color=material
            #make rgb into hsv
            h,s,v= cv.cvtColor(np.uint8([[[material_color[2][0], material_color[2][1], material_color[2][2]]]]), cv.COLOR_RGB2HSV)[0][0]
            material_colors_img= make_material_image(material[1], h, s, v, material_colors_img) # mask h s v image
    
    hsv_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2HSV)

    # extract and Flatten the hue channel
    hue_channel = hsv_img[:,:,0]

    hue_values = hue_channel.flatten()
    #remove black pixels from the hue values using the value channel
    val_channel = hsv_img[:,:,2]
    val_values = val_channel.flatten()
    sat_values = hsv_img[:,:,1].flatten()
    hue_values, sat_values, val_values= remove_black_zeros(hue_values, sat_values, val_values)
    #total pixels is length of hue values
    total_pixels_img= len(hue_values)


    hue_peaks, left_bases, right_bases= calc_hue_kde(hue_values, total_pixels_img)


    #itterate for each hue found
    if show_graphs:
        print('There are ',len(hue_peaks),'Hue peaks')
   
    for hue_material_number in range(len(hue_peaks)):

        #mask for object segmentation
        lower_mask = hsv_img[:,:,0] > left_bases[hue_material_number] 
        upper_mask = hsv_img[:,:,0] < right_bases[hue_material_number]
        maskhue = upper_mask*lower_mask

        #show segmented image
        masked_img = make_new_image(rgb_img, maskhue)
        maskhue_without_noise= remove_small_objects(maskhue, 40)
        masked_img2 = make_new_image(rgb_img, maskhue_without_noise)
        #if masked image has less that 15 pixels, skip
        n_pixels= count_pixels_of_image(masked_img)

        n_pixels2= count_pixels_of_image(masked_img2)
        if n_pixels < 15:
            if show_graphs:
                print('invalid material due to few pixels in hue loop:', n_pixels2)
            continue
        if show_graphs:
            #show masked images
            plt.figure()
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            plt.title('Masked image')
            masked_img_display= display_in_green(masked_img)
            plt.imshow(masked_img_display)
            plt.axis('off')
            #plt.xlabel('npixels: '+ str(n_pixels))
            plt.subplot(1, 2, 2) # index 2
            plt.title('Masked image without noise')
            plt.axis('off')
            masked_img_display= display_in_green(masked_img2)
            plt.imshow(masked_img_display)
            #plt.xlabel('npixels: '+ str(n_pixels2))
            

        extract_color_new(Specular, material_properties,0,hue_material_number, maskhue_without_noise,masked_img2, rgb_img , material_colors_img , hue_peaks[hue_material_number],show_graphs)
    return material_colors_img, material_properties

def extract_color_value(Specular,material_properties,hue_material_number,masked_img , material_colors_img , hue,saturation, show_graphs=False):
    '''hue= int(hue)
    saturation= int(saturation)'''

    masked_hsv_img = cv.cvtColor(masked_img, cv.COLOR_RGB2HSV)
    total_pixels_maskedv= count_pixels_of_image(masked_hsv_img)
    #total_pixels_maskedv= int(total_pixels_maskedv)

    if total_pixels_maskedv <= 10:
        if show_graphs:
            print('invalid material due to few pixels in value:', total_pixels_maskedv)
        hue_material_number ='N'
        return
    

    val_channel = masked_hsv_img[:,:,2]
    val_values = val_channel.flatten()
    #check if its finite 
    if np.all(np.isfinite(val_values)) == False:
        if show_graphs:
            print('invalid material due to non finite values')
        hue_material_number ='N'
        return

    hist_v,v_end, segmentation= calc_val_kde(val_values, total_pixels_maskedv, 0.90)
    print(segmentation)
    starts= segmentation[:,0]
    ends= segmentation[:,1]
    if show_graphs:
        plt.figure()
        plt.plot(hist_v)
        plt.scatter(v_end,np.zeros(len(v_end)), label='Chosen value', color='blue', marker='o')
        #start and end of segmentation
        plt.scatter(starts, np.zeros(len(starts)), color='red', marker='>', label='Segmentation start')
        plt.scatter(ends, np.zeros(len(ends)), color='red', marker='<' , label='Segmentation end')
        #plt.title('KDE of value with 90% of pixels')
        plt.xlabel('Value')
        plt.ylabel('Number of pixels')
        plt.legend()
        plt.show()
    
    for i in range(0, len(v_end)):
        hue_material_number=0
        v_end[i]= int(v_end[i])
        value= v_end[i]
        #create new mask
        lower_mask = masked_hsv_img[:,:,2] >= segmentation[i,0]
        upper_mask = masked_hsv_img[:,:,2] <= segmentation[i,1]
        maskvnew = upper_mask*lower_mask
        masked_imgnew= make_new_image(masked_img, maskvnew)
        total_pixels_maskedv= count_pixels_of_image(masked_imgnew)


        if total_pixels_maskedv < 10:
            if show_graphs:
                print('invalid material due to few pixels in value:', total_pixels_maskedv)
            hue_material_number ='N'
            return

        if hue_material_number !='N':
            final_color_RGB= print_color_from_hsv(hue,saturation,value,size=50)

            specular_data= extract_specular_data(Specular, maskvnew)

            if show_graphs:
                plt.figure()
                plt.subplot(1, 2, 1) # row 1, col 2 index 1
                #plt.title('Final colour of object R1')
                masked_img_display= display_in_green(masked_imgnew)
                plt.imshow(masked_img_display)
                plt.axis('off')

                plt.subplot(1, 2, 2) # index 2
                plt.imshow(final_color_RGB)
                plt.axis('off')

                #plt.title('specular: '+ str(specular_data))
                #plt.xlabel('npixels: '+ str(total_pixels_maskedv))

            material_colors_img = make_material_image(masked_imgnew, hue, saturation,value, material_colors_img)

            if show_graphs:
                print('number of pixels of object above', total_pixels_maskedv,', HSV', hue, saturation,value)
                print('specular data', specular_data)
        
            material_color_hsv= np.uint8([[[hue,saturation,value]]])  
            material_color_rgb = cv.cvtColor(material_color_hsv, cv.COLOR_HSV2RGB)
            # mat_number, where it acts, rgb alpha, enabled?
            material_properties.append([total_pixels_maskedv,maskvnew,[material_color_rgb[0][0][0], material_color_rgb[0][0][1], material_color_rgb[0][0][2]], specular_data])


def extract_color_new(Specular,material_properties, recurson,hue_material_number, maskec,masked_rgb_img, rgb_img , material_colors_img , hue, show_graphs=False):
                                                                                                                                                      
    masked_hsv_img = cv.cvtColor(masked_rgb_img, cv.COLOR_RGB2HSV)
    total_pixels_maskedec= count_pixels_of_image(masked_hsv_img)
    if total_pixels_maskedec <= 10:
        if show_graphs:
            print('invalid material due to few pixels:', total_pixels_maskedec)
        hue_material_number ='N'
        return

   
    sat_channel = masked_hsv_img[:,:,1]
    sat_values = sat_channel.flatten()

    val_channel = masked_hsv_img[:,:,2]
    val_values = val_channel.flatten()

    
    hist_s,s_start_end = calc_sat_kde(sat_values, total_pixels_maskedec, True)
        
    if show_graphs:
        print('total masked pixels in EC ',total_pixels_maskedec)

    if (len(s_start_end)<=2):
        if len(s_start_end) >=1 or recurson==1 :
            if show_graphs:
                print('there is only one distinct peak in saturation')
            saturation= s_start_end[-1]
        if len(s_start_end)==0:
            if show_graphs:
                print('sat invalid so made zero ')
            hue_material_number ='N'
            saturation= 0
       
        #this is where the value function should go
        
        hist_v,v_end, segmentation = calc_val_kde(val_values, total_pixels_maskedec, 0.90)
        if show_graphs:
            starts= segmentation[:,0]
            ends= segmentation[:,1]
            plt.figure()
            plt.plot(hist_v)
            plt.scatter(v_end,np.zeros(len(v_end)), label='Chosen value', color='blue', marker='o')
            #start and end of segmentation
            plt.scatter(starts, np.zeros(len(starts)), color='red', marker='>', label='Segmentation start')
            plt.scatter(ends, np.zeros(len(ends)), color='red', marker='<' , label='Segmentation end')
            #plt.title('KDE of value with 90% of pixels')
            plt.xlabel('Value')
            plt.ylabel('Number of pixels')
            plt.legend()
            plt.show()
        
        for i in range(0, len(v_end)):
            hue_material_number=0
            v_end[i]= int(v_end[i]) 
            value= v_end[i]
            #create new mask
            lower_mask2 = masked_hsv_img[:,:,2] >= segmentation[i,0]
            upper_mask2 = masked_hsv_img[:,:,2] <= segmentation[i,1]
            maskvnew2 = upper_mask2*lower_mask2
            masked_rgb_img2= make_new_image(masked_rgb_img, maskvnew2)
            total_pixels_maskedec= count_pixels_of_image(masked_rgb_img2)


            if total_pixels_maskedec < 10:
                if show_graphs:
                    print('invalid material due to few pixels in value:', total_pixels_maskedec)
                hue_material_number ='N'
                return

            if hue_material_number !='N':
                final_color_RGB= print_color_from_hsv(hue,saturation,value,size=50)
                specular_data= extract_specular_data(Specular, maskvnew2)

                if show_graphs:
                    plt.figure()
                    plt.subplot(1, 2, 1) # row 1, col 2 index 1
                    plt.title('Final segmentation of object')
                    masked_image_display= display_in_green(masked_rgb_img2)
                    
                    plt.imshow(masked_image_display)
                    plt.axis('off')

                
                    plt.subplot(1, 2, 2) # index 2
                    plt.title('Final colour of object')

                    plt.imshow(final_color_RGB)
                    plt.axis('off')

                    #plt.title('specular: '+ str(specular_data))
                    #plt.xlabel('npixels: '+ str(total_pixels_maskedec))

                material_colors_img = make_material_image(masked_rgb_img2, hue, saturation,value, material_colors_img)

                if show_graphs:
                    print('number of pixels of object above', total_pixels_maskedec,', HSV', hue, saturation,value)
                    print('specular data', specular_data)
        
                material_color_hsv= np.uint8([[[hue,saturation,value]]])  
                material_color_rgb = cv.cvtColor(material_color_hsv, cv.COLOR_HSV2RGB)
                # mat_number, where it acts, rgb alpha, enabled?
                material_properties.append([total_pixels_maskedec,maskvnew2,[material_color_rgb[0][0][0], material_color_rgb[0][0][1], material_color_rgb[0][0][2]], specular_data])

    if (len(s_start_end)>2):
       
        if show_graphs:
            print('there are several distinct peaks in saturation', len(s_start_end)//2)
        hue_material_number ='N'
        # calculate new mask ans apply
        for i in range(0, len(s_start_end)//2):
            if show_graphs:
                print('these are the saturation ranges'+ str(s_start_end))
            lower_mask = masked_hsv_img[:,:,1] >= s_start_end[2*i] 
            upper_mask = masked_hsv_img[:,:,1] <= s_start_end[2*i + 1]
            maskec = upper_mask*lower_mask # this is a new mask
            #clean mask 
            noisy_image= make_new_image(masked_rgb_img, maskec)
            total_pixels_maskedec= count_pixels_of_image(noisy_image)
            if show_graphs:
                print('this is how many pixels are in old mask in EC', total_pixels_maskedec)
            maskecnew= remove_small_objects(maskec, 40)
            new_image= make_new_image(masked_rgb_img, maskecnew) # for making a rgb image  becomes masked image
            npixelsbefore= total_pixels_maskedec
            total_pixels_maskedecnew= count_pixels_of_image(new_image)
            if show_graphs:
                print('total pixels in new mask in EC', total_pixels_maskedecnew)
            saturation= s_start_end[2*i + 1]
            # print('this is the new masked image')
            if show_graphs:
                plt.figure()
                plt.subplot(1, 2, 1) # row 1, col 2 index 1
                #plt.title('noisy Masked Image')
                noisy_image_display= display_in_green(noisy_image)
                plt.imshow(noisy_image_display)
                plt.axis('off')
                #plt.xlabel('pixels before'+str(npixelsbefore))
                plt.subplot(1, 2, 2) # index 2
                masked_img_display= display_in_green(new_image)
                plt.imshow(masked_img_display)
                plt.axis('off')
                #plt.xlabel('pixels'+str(total_pixels_maskedecnew))
                #plt.title('this is the new mask with sat ranges ' + str(s_start_end[2*i]) +' '+ str(s_start_end[2*i + 1]))
                #def extract_color_value(Specular,material_properties, recurson,hue_material_number, maskv,masked_img, rgb_img , material_colors_img , hue,saturation, show_graphs=False):

                extract_color_value(Specular, material_properties,0 ,new_image , material_colors_img ,hue,saturation, show_graphs=True)
            if show_graphs==0:
                extract_color_value(Specular, material_properties,0 ,new_image , material_colors_img ,hue,saturation, show_graphs=False)

            

    if(show_graphs and hue_material_number !='N'):
        display_hue(hue)
        
    

def show_material_img(material_colors_img):
    #convert from hsv to rgb
    material_colors_img_rgb = cv.convertScaleAbs(material_colors_img)
    material_colors_img_rgb = cv.cvtColor(material_colors_img_rgb, cv.COLOR_HSV2RGB)

    #show the material colors
    plt.figure()
    plt.imshow(material_colors_img_rgb)
    plt.axis('off')
    #plt.title('Material colours')
    plt.show()


def remove_small_objects(mask, min_size):
   
    #make sure mask is in binary
    mask = mask.astype(np.uint8)
    #find all the contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #create a new mask
    new_mask = np.zeros_like(mask)
    #mask size 
    mask_size= np.sum(mask)
    min_size= mask_size*0.005
    #for each contour
    for contour in contours:
        #if the contour is large enough
        if cv.contourArea(contour) > min_size:
            #draw the contour on the new mask
            cv.drawContours(new_mask, [contour], -1, 255, -1)
    new_mask=cv.bitwise_and( new_mask,mask)
    new_mask = new_mask.astype(np.uint8)
    return new_mask