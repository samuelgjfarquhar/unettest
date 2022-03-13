import numpy as np
import nibabel as nib
import random
import glob
from tensorflow.keras.utils import to_categorical
<<<<<<< HEAD
=======
import splitfolders  
import matplotlib.pyplot as plt
from tifffile import imsave
>>>>>>> 2496f27ad8d9095f486bc8552883a480383234c2
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import torchio as tio

scaler = MinMaxScaler()

<<<<<<< HEAD
liver_list = sorted(glob.glob('D:\\liver\\test\\train\\*.nii'))
=======
TRAIN_DATASET_PATH = "C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\"
test_image_flair = nib.load(TRAIN_DATASET_PATH + "BraTS20_Training_355\\BraTS20_Training_355_flair.nii").get_fdata()
test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)
>>>>>>> 2496f27ad8d9095f486bc8552883a480383234c2

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
    combined_images = skTrans.resize(combined_images, (128,128,256), order=1, preserve_range=True)
    combined_mask = skTrans.resize(combined_mask, (128,128,256), order=1, preserve_range=True)  
        
    np.save('D:\\liver\\test\\npyTrain\\liver_'+str(img)+'.npy', combined_images)
    np.save('D:\\liver\\test\\npyMask\\liver_'+str(img)+'.npy', combined_mask)

    print(f'img {img} converted')


<<<<<<< HEAD
# img = np.load('D:\\liver\\test\\npyTrain\\liver_1.npy')
# mask = np.load('D:\\liver\\test\\npyMask\\liver_1.npy')

# plt.subplot(234)
# plt.imshow(img[:,:,90], cmap='gray')
# plt.title('Image t2')
# plt.subplot(235)
# plt.imshow(mask[:,:,90], cmap='gray')
# plt.title('Mask')
# plt.show()

# testimg = np.load('D:\\liver\\test\\npyTrain\\testliver_1.npy')
# testmask = np.load('D:\\liver\\test\\npyMask\\testliver_1.npy')

# plt.subplot(234)
# plt.imshow(testimg[:,:,90], cmap='gray')
# plt.title('Image t2')
# plt.subplot(235)
# plt.imshow(testmask[:,:,90], cmap='gray')
# plt.title('Mask')
# plt.show()

=======
test_mask[test_mask==4] = 3  #Reassign mask values 4 to 3


n_slice=random.randint(0, test_mask.shape[2])

combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)
combined_x=combined_x[56:184, 56:184, 13:141] #Crop to 128x128x128x4
test_mask = test_mask[56:184, 56:184, 13:141]

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(combined_x[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

imsave('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\combined\\combined255.tif', combined_x)
np.save('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\combined\\combined255.npy', combined_x)

my_img=np.load('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\combined\\combined255.npy')

test_mask = to_categorical(test_mask, num_classes=4)

#t1_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
t2_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*\\*t2.nii'))
t1ce_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*\\*t1ce.nii'))
flair_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*\\*flair.nii'))
mask_list = sorted(glob.glob('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*\\*seg.nii'))

for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)
      
    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
    
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
   
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.01:  
        print("Save Me")
        temp_mask= to_categorical(temp_mask, num_classes=4)
        np.save('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\input_data_3channels\\images\\image_'+str(img)+'.npy', temp_combined_images)
        np.save('C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\input_data_3channels\\masks\\mask_'+str(img)+'.npy', temp_mask)
        
    else:
        print("I am useless")

input_folder = 'C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\input_data_3channels'
output_folder = 'C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\input_data_128'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None) 
>>>>>>> 2496f27ad8d9095f486bc8552883a480383234c2
