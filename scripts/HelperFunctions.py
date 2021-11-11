'''
This file contains declarations of important functins to be used in the main notebook, it contains all the necessary libraries imports. 

Authors : Youssef SNOUSSI & SADIKI Nour Ed-dine
'''

import os
from pydicom.filereader import dcmread
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sn
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import figure_factory as FF
from plotly.graph_objs import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Function to load the images

def load_images(path):
    
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x : int(x.InstanceNumber))
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slice[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SlicesThicknes = slice_thickness
        
    return slices

# Function to convert the pixel values to the corresponding Hounsfield units

def get_hu(scans):
    
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    image[image == -2000] == 0
    
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

# Function to show the images

def img_stack(imgs, rows=7, cols=7, start_with=1, step=3):
    
    fig, ax = plt.subplots(rows, cols, figsize=[24,24])
    
    for i in range(rows * cols):
        ind = start_with + i * step
        ax[int(i / rows), int(i % rows)].set_title("slice %d" % ind)
        ax[int(i / rows), int(i % rows)].imshow(imgs[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    
    plt.show()
    
    
# Functions to resample images

def resample(image, scan, new_spacing=[1,1,1]):
    
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = sn.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

# Ploting interactive 3D 
def plotly_3d(verts, faces):
    
    x,y,z = zip(*verts)
    print("Drawing")
    colormap = ['rgb(236, 236, 212)','rgb(236, 236, 212)']
    fig = FF.create_trisurf(x=x,
                           y=y,
                           z=z,
                           plot_edges=False,
                           colormap=colormap,
                           simplices=faces,
                           backgroundcolor='rgb(64,64,64)',
                           title='Interactive Visualization')
    iplot(fig)
    
# Ploting 3D
def plt_3d(verts, faces):
    
    print ("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [0.4, 0.5, 0.6]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()
    
    
# Segmentation

def make_mask(img, display=False):
    
    row_size = img.shape[0]
    col_size = img.shape[1]
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[int(col_size/5):int(col_size/5*4), int(row_size/5):int(row_size/5*4)]
    mean = np.mean(middle)
    maxe = np.mean(img)
    mine = np.mean(img)
    
    img[img==maxe] = mean
    img[img==mine] = mean
    
    
# on uilise KMeans pour effectuer un clustering sur les images 
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold,1.0,0.0)
    
    eroded = morphology.erosion(thresh_img, np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))
    
    
# Etiquiter les differentes regions dans l'image
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0] < row_size/10*9 and B[3]-B[1] < col_size/10*9 and B[0] > row_size/5 and B[2] < col_size/5*4:
            good_labels.append(prop.label)
    
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0
    
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) 
    
    
# Affichage des image, treshhold, labeles images et les masks


    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img