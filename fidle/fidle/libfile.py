import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from skimage import io, color, exposure, transform
import h5py


# def func(x):
#     return 2*x


def plot_images(images='None',labels='None',figsize=(25,15)):
    
    # make a fiture and compute its columns and rows:
    fig = plt.figure(figsize=figsize)
    columns = 9
    imnumb = len(images)
    rows = np.int(imnumb/columns) + (imnumb-np.int(imnumb/columns)*columns >= 1)
    
    # plot all images in figure :
    i=1
    for image in images: 
        #Add a subplot at the i(th) position, show on the i(th) image, show off the axis and put the title:
        fig.add_subplot(rows,columns,i)
        plt.imshow(image)
        plt.axis('off')
        plt.title(labels[i-1])
        i=i+1




def plot_images_rand(images='None',labels='None',subset_im_numb=50, figsize=(25,18),pred='None'):
    # choose randomly subset from images:
    img_subset_index=random.randint(len(images)-1, size=(subset_im_numb))
    # make a fiture and compute its columns and rows:
    fig = plt.figure(figsize=figsize)
    columns = 9
    imnumb = subset_im_numb
    rows = np.int(imnumb/columns) + (imnumb-np.int(imnumb/columns)*columns >= 1)
    
    # plot all images in figure :
    i=1
    for img_index in img_subset_index:
        #Add a subplot at the i(th) position, show on the i(th) image, show off the axis and put the title:
        fig.add_subplot(rows,columns,i)
        plt.imshow(images[img_index])
        plt.axis('off')
        # make a tile:
        if pred != 'None' and pred[img_index] != labels[img_index]:
            title = f'{labels[img_index]}({pred[img_index]})'
        else:
            title = labels[img_index] 
        plt.title(title)
        #plt.title(labels[img_index])
        #plt.title(labels[img_index])
        i=i+1


# def plot_pred_images_rand(model='None',images='None',labels='None',subset_im_numb=50, figsize=(25,18)):
#     # pick up randomly a subset from images:
#     img_subset=random.randint(len(images)-1, size=(subset_im_numb))
    
#     # predict on the random subset:
#     y_sigmoid = model.predict(img_subset)
#     y_pred  = np.argmax(y_sigmoid, axis=-1)
    
#     # make a fiture and compute its columns and rows:
#     fig = plt.figure(figsize=figsize)
#     columns = 9
#     imnumb = subset_im_numb
#     rows = np.int(imnumb/columns) + (imnumb-np.int(imnumb/columns)*columns >= 1)
    
#     # plot all images in figure :
#     i=1
#     for img_index in img_subset:
#         #Add a subplot at the i(th) position, show on the i(th) image, show off the axis and put the title:
#         fig.add_subplot(rows,columns,i)
#         plt.imshow(images[img_index])
#         plt.axis('off')
#         # make a tile:
#         if pred[img_index] == labels[img_index]:
#             title = labels[img_index]
#         else:
#             title = f'{labels[img_index]}({pred[img_index]})'
#         plt.title(title)
#         plt.title(labels[img_index])
#         i=i+1


def cook_images(images, width, height,mode='RGB'):
    # receive images and rturn images with required mod
    cooked_images=[]
    modes = {'RGB':3,'RGB-HE':3,'L':1,'L-HE':1,'L-LHE':1,'L-CLAHE':1}
    lz = modes[mode]
    for image in images:
        # from rgba ro rgb 
        if image.shape[2]==4:
            image=color.rgba2rgb(image)
        # resize:
        image = transform.resize(image, (width,height))
        # ---- RGB / Histogram Equalization
        if mode=='RGB-HE':
            hsv = color.rgb2hsv(image.reshape(width,height,3))
            hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
            img = color.hsv2rgb(hsv)
        # to grayscale
        if mode =='L':
            image=color.rgb2gray(image)
            
        # to gray + histor equalizer
        if mode == 'L-HE':
            image=color.rgb2gray(image)
            image=exposure.equalize_hist(image)
        # ---- Grayscale / Local Histogram Equalization
        if mode=='L-LHE':        
            image=color.rgb2gray(image)
            image = img_as_ubyte(image)
            image=rank.equalize(image, disk(10))/255.
        
        # ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)
        if mode=='L-CLAHE':
            image=color.rgb2gray(image)
            image=exposure.equalize_adapthist(image)
            
        cooked_images.append(image)
        pwk.update_progress('Enhancement: ',len(cooked_images),len(images))
    
    #
    cooked_images = np.array(cooked_images,dtype='float64')
    cooked_images = cooked_images.reshape(-1,width,height,lz)
    return cooked_images

    
    
def h5_save_data(x_train='None',y_train='None',x_test='None',y_test='None',x_meta='None',y_meta='None',filename='None'):
    
    with h5py.File(filename,"w") as f:
        f.create_dataset("x_train",data=x_train)
        f.create_dataset("y_train",data=y_train)
        f.create_dataset("x_test",data=x_test)
        f.create_dataset("y_test",data=y_test)
        f.create_dataset("x_meta",data=x_meta)
        f.create_dataset("y_meta",data=y_meta)
     
    
def  rescale_rand(data,scale=0.2):
    
    # tak scale% randomly from data:
    size = int(scale*len(data))
    #img_index=np.random.randint(len(data)-1, size=size)
    #return np.array([data[indx] for indx in img_index])
    return data[:size]
    
# # Example for using subplot:   
# image = io.imread(f'{data_set_dir}/datasets/GTSRB/origine/Meta/0.png')
# # make a figure:
# fig = plt.figure(figsize=(5,5))
# rows, columns = 2,2
# # Add  subplot at 1st position, In the 1st position, show on an image, show off the axis and put the title:
# fig.add_subplot(rows,columns,1)
# plt.imshow(image)
# plt.axis('off')
# plt.title('1st image')
# # Add  subplot at 1st position, In the 1st position, show on an image, show off the axis and put the title:
# fig.add_subplot(rows,columns,2)
# plt.imshow(image)
# plt.axis('off')
# plt.title('2nd image')
# # Add  subplot at 1st position, In the 1st position, show on an image, show off the axis and put the title:
# fig.add_subplot(rows,columns,3)
# plt.imshow(image)
# plt.axis('off')
# plt.title('3d image')
# # Add  subplot at 1st position, In the 1st position, show on an image, show off the axis and put the title:
# fig.add_subplot(rows,columns,4)
# plt.imshow(image)
# plt.axis('off')
# plt.title('4th image')
