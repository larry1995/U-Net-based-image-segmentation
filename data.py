from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_uint
from skimage import img_as_ubyte

from PIL import Image
from skimage.io import imread
import cv2




def histoequalization(image_path,num_image):
    for i in range(num_image):
        img = io.imread(os.path.join(image_path,"%d.tif"%i),as_gray=True)
        hist,bins = np.histogram(img.flatten(),256,[0,256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
#plt.subplot(2,4,1)
#plt.imshow("original",img)
#plt.subplot(1,3,1)
#plt.hist(img.flatten(),256,[0,256], color = 'r')
#plt.xlim([0,256])
#plt.legend(('cdf','histogram'), loc = 'upper left')

        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        img2 = cdf[img]
        print("image normalized")
        cv2.imwrite('data/membrane/train/raw_histo/%d.tif'%i,img2)
        #io.imsave(os.path.join(save_path,"%d.tif"%i),result_img)
        #plt.subplot(1,3,2)
        #plt.imshow(img2)
        #plt.subplot(1,3,3)
    return img2 


def adjustData(img,mask,flag_multi_class,num_class): ##input
    if(flag_multi_class):
        img = img / 255.0
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255.0
        mask = mask /255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

#def image_normalized(dir_path,save_dir):
    '''
    tif£¬size:512*512£¬gray
    :param dir_path: path to your images directory
    :param save_dir: path to your images after normalized
    :return:
    '''
    #for file_name in os.listdir(dir_path):
        #if os.path.splitext(file_name)[1].replace('.', '') == "PNG":
            #jpg_name = os.path.join(dir_path, file_name)
            #save_path = os.path.join(save_dir,file_name)
            #img = cv2.imread(jpg_name, cv2.COLOR_RGB2GRAY)
            #img_standard = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            #img_standard = cv2.cvtColor(img_standard, cv2.COLOR_BGR2GRAY)
            #cv2.imwrite(save_path, img_standard)

        

    
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def validationGenerator(batch_size,vali_path,image_folder,mask_folder,image_color_mode = "grayscale",target_size=(512,512)):
    #vali_image_datagen = ImageDataGenerator(rotation_range=0.2)
    #vali_mask_datagen = ImageDataGenerator(rotation_range=0.2)
    vali_image_datagen = ImageDataGenerator(rescale = 1./255)
    vali_mask_datagen = ImageDataGenerator(rescale=1./255)
    vali_image_generator = vali_image_datagen.flow_from_directory(
        vali_path,
        classes = [image_folder],
        target_size = target_size,
        batch_size = batch_size,
        class_mode = None,
        color_mode = image_color_mode
        )
    vali_mask_generator = vali_mask_datagen.flow_from_directory(
        vali_path,
        classes = [mask_folder],
        target_size = target_size,
        batch_size = batch_size,
        class_mode = None,
        color_mode = image_color_mode
        )
    vali_generator = zip(vali_image_generator,vali_mask_generator)
    for (vali_img,vali_mask) in vali_generator:
        #vali_img,vali_mask = adjustData(vali_img,vali_mask,flag_multi_class =False,num_class = 2)
        yield(vali_img,vali_mask)

#def validationGenerator(validation_path,mask_path,num_image =22,target_size = (512,512),flag_multi_class = False,as_gray = True):
    #for i in range(num_image):
        #img = io.imread(os.path.join(validation_path,"%d.png"%i),as_gray = as_gray)
        #img = img / 255.0
        #img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        #img = np.reshape(img,(1,)+img.shape)
    #for j in range(num_image):
       #mask = io.imread(os.path.join(mask_path,"%d.png"%j),as_gray = as_gray)
        #mask = mask / 255.0
        #mask = trans.resize(mask,target_size)
        #mask = np.reshape(mask,mask.shape+(1,)) if (not flag_multi_class) else mask
        #mask = np.reshape(mask,(1,)+mask.shape)
        
        #yield (img,mask)


    

def testGenerator(test_path,num_image =23,target_size = (512,512),flag_multi_class = False,as_gray = True):
    #test_datagen = ImageDataGenerator(rescale=1./255)
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.tif"%i),as_gray = as_gray)
        #img = img / 255.0
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255.0



def saveResult(save_path,npyfile,size=(512,512),flag_multi_class = False,num_class = 2,threshold=1256):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img = img_as_ubyte(img)
        #img_at_mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,10,10)
        print("Images created")
        

        io.imsave(os.path.join(save_path,"%d_predict.png"%i), img_as_uint(img)) ##save csv
        #io.imsave(os.path.join(save_path,"%d_predict.tif"%i), img_as_btype(img)) 
def post_processing(image_path,save_path,img_num):
    for i in range(img_num):
    #img = io.imread(os.path.join("C:\\Users\\hcntb\\Desktop\\testdta_0.5_0.4_10_150\\testdata","%d_%s.png"%(i,"predict")),as_gray=True)
        img = io.imread(os.path.join(image_path,"%d_%s.png"%(i,"predict")),as_gray=True)
        img=img_as_ubyte(img)
    #gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, result_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        io.imsave(os.path.join(save_path,"%d_result.png"%i),result_img)
print("The images are processed")
