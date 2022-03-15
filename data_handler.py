import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import torchio as tio
import skimage.transform as skTrans
scaler = MinMaxScaler()

liver_list = sorted(glob.glob('D:\\liver\\train\\*.nii'))
mask_list = sorted(glob.glob('D:\\liver\\mask\\*.nii'))

#Training numpy conversion
for img in range(len(liver_list)):   #Using t1_list as all lists are of same size
   
   
    val_liver=nib.load(liver_list[img]).get_fdata()
    val_liver=scaler.fit_transform(val_liver.reshape(-1, val_liver.shape[-1])).reshape(val_liver.shape)
           
    val_mask=nib.load(mask_list[img]).get_fdata()
    val_mask=val_mask.astype(np.uint8)
    
    combined_images =val_liver[177:443, 135:391, 340:596]
    combined_mask = val_mask[177:443, 135:391, 340:596]
    
    combined_images = skTrans.resize(combined_images, (128,128,128), order=1, preserve_range=True)
    combined_mask = skTrans.resize(combined_mask, (128,128,128), order=1, preserve_range=True)  
        
    temp_liver = np.stack([combined_images], axis=3)
    temp_mask= to_categorical(combined_mask, num_classes=3)
    
    np.save('D:\\test\\ttT\\liver_'+str(img)+'.npy', temp_liver)
    np.save('D:\\test\\ttM\\liver_'+str(img)+'.npy', temp_mask)

    print(f'img {img} converted')


#################################

#Training numpy conversion
for img in range(0,2):   #Using t1_list as all lists are of same size
   
   
    val_liver=nib.load(liver_list[img]).get_fdata()
    val_liver=scaler.fit_transform(val_liver.reshape(-1, val_liver.shape[-1])).reshape(val_liver.shape)
           
    val_mask=nib.load(mask_list[img]).get_fdata()
    val_mask=val_mask.astype(np.uint8)
    
    combined_images =val_liver[177:443, 135:391, 340:596]
    combined_mask = val_mask[177:443, 135:391, 340:596]
    
    combined_images = skTrans.resize(combined_images, (128,128,128), order=1, preserve_range=True)
    combined_mask = skTrans.resize(combined_mask, (128,128,128), order=1, preserve_range=True)  
        
    temp_liver = np.stack([combined_images], axis=3)
    temp_mask= to_categorical(combined_mask, num_classes=3)
    
    np.save('D:\\test\\ttT\\liver_'+str(img)+'.npy', temp_liver)
    np.save('D:\\test\\ttM\\liver_'+str(img)+'.npy', temp_mask)

    print(f'img {img} converted')



val_liver=nib.load(liver_list[1]).get_fdata()
val_liver=scaler.fit_transform(val_liver.reshape(-1, val_liver.shape[-1])).reshape(val_liver.shape)
           
val_mask=nib.load(mask_list[1]).get_fdata()
val_mask=val_mask.astype(np.uint8)

combined_images =val_liver[177:443, 135:391, 340:596]
combined_mask = val_mask[177:443, 135:391, 340:596]

combined_images = skTrans.resize(combined_images, (128,128,128), order=1, preserve_range=True)
combined_mask = skTrans.resize(combined_mask, (128,128,128), order=1, preserve_range=True)  
    
temp_liver = np.stack([combined_images], axis=3)
temp_mask= to_categorical(combined_mask, num_classes=3)
    
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('combined image')
plt.imshow(combined_mask[:,:,100])
plt.subplot(232)
plt.title('to categorical')
plt.imshow(temp_mask[:,:,100])
plt.show()
#