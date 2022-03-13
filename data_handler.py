import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import torchio as tio

scaler = MinMaxScaler()

liver_list = sorted(glob.glob('D:\\liver\\test\\train\\*.nii'))

mask_list = sorted(glob.glob('D:\\liver\\test\\mask\\*.nii'))

#Training numpy conversion
for img in range(len(liver_list)):   #Using t1_list as all lists are of same size
   
   
    val_liver=nib.load(liver_list[img]).get_fdata()
    val_liver=scaler.fit_transform(val_liver.reshape(-1, val_liver.shape[-1])).reshape(val_liver.shape)
           
    val_mask=nib.load(mask_list[img]).get_fdata()
    val_mask=val_mask.astype(np.uint8)
        
    combined_images =val_liver[177:443, 135:391, 340:596]
    combined_mask = val_mask[177:443, 135:391, 340:596]
    
    import skimage.transform as skTrans
    combined_images = skTrans.resize(combined_images, (128,128,128), order=1, preserve_range=True)
    combined_mask = skTrans.resize(combined_mask, (128,128,128), order=1, preserve_range=True)  
        
    np.save('D:\\liver\\test\\npyTrain\\liver_'+str(img)+'.npy', combined_images)
    np.save('D:\\liver\\test\\npyMask\\liver_'+str(img)+'.npy', combined_mask)

    print(f'img {img} converted')


img = np.load('D:\\liver\\test\\npyTrainTest\\liver_1.npy')
mask = np.load('D:\\liver\\test\\npyMaskTest\\liver_1.npy')

plt.subplot(234)
plt.imshow(img[:,:,50], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(mask[:,:,50], cmap='gray')
plt.title('Mask')
plt.show()

testimg = np.load('D:\\liver\\test\\npyTrain\\liver_1.npy')
testmask = np.load('D:\\liver\\test\\npyMask\\liver_1.npy')

plt.subplot(234)
plt.imshow(testimg[:,:,100], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(testmask[:,:,100], cmap='gray')
plt.title('Mask')
plt.show()

