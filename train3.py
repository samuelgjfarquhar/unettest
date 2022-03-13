import os
import numpy as np
from data_generator import imageLoader
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random
from unet_arch import unet_model
import pandas as pd
import segmentation_models_3D as sm
from keras.metrics import MeanIoU
from matplotlib import pyplot as plt
import random
from optparse import OptionParser
import tensorflow_addons as tfa

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.chdir('C:/Users/samue/BraTS2020/')

####################################################
train_img_dir = "D:/liver/test/npyTrainTest/"
train_mask_dir = "D:/liver/test/npyMaskTest/"

val_img_dir = "D:/liver/test/npyValTest/"
val_mask_dir = "D:/liver/test/npyValMaskTest/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
batch_size = 1

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list,
                              val_mask_dir, val_mask_list, batch_size)

img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(223)
plt.imshow(test_img[:,:,n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

wt0, wt1, wt2 = 0.26, 22.53, 22.53

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

model = unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=128,
                          IMG_CHANNELS=1,
                          num_classes=3)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)

history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=1,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    )

model.save('brats_3d.hdf5')
##################################################################


# plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#################################################
from keras.models import load_model

# Load model for prediction or continue training

# For continuing training....
# The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
# This is because the model does not save loss function and metrics. So to compile and
# continue training we need to provide these as custom_objects.
my_model = load_model('brats_3d.hdf5')

# So let us add the loss as custom object... but the following throws another error...
# Unknown metric function: iou_score
my_model = load_model('brats_3d.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss})

# Now, let us add the iou_score function we used during our initial training
my_model = load_model('brats_3d.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score': sm.metrics.IOUScore(threshold=0.5)})

# Now all set to continue the training process.
history2 = my_model.fit(train_img_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=1,
                        verbose=1,
                        validation_data=val_img_datagen,
                        validation_steps=val_steps_per_epoch,
                        )
#################################################

# For predictions you do not need to compile the model, so ...
my_model = load_model('brats_3d.hdf5',
                      compile=False)

# Verify IoU on a batch of images from the test dataset
# Using built in keras function for IoU
# Only works on TF > 2.0
from keras.metrics import MeanIoU

batch_size = 4  # Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list,
                               val_mask_dir, val_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

#val_loss, val_dice = model.evaluate(val_img_datagen)
#print(f"validation soft dice loss: {val_loss:.4f}")
#print(f"validation dice coefficient: {val_dice:.4f}")

#n_classes = 4
#IOU_keras = MeanIoU(num_classes=n_classes)
#IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
#print("Mean IoU =", IOU_keras.result().numpy())

#brain val
val_num = 12
test_val = np.load("C:\\Users\\samue\\BraTS2020\\BraTS2020_TrainingData\\input_data_3channels\\images\\image_" + str(val_num) + ".npy")

test_val_img_input = np.expand_dims(test_val, axis=0)
test_val_prediction = my_model.predict(test_val_img_input)
test_val_prediction_argmax = np.argmax(test_val_prediction, axis=4)[0, :, :, :]

n_slice_val=70

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Validation Test Image')
plt.imshow(test_val[:, :, n_slice_val, 1], cmap='gray')
plt.subplot(232)
plt.title('Prediction on validation image')
plt.imshow(test_val_prediction_argmax[:, :, n_slice_val])
plt.show()


#liver TEST val###############################

test_liver = np.load("D:\\MDS\\dataset\\Task03_Liver\\numpy\\val\\liverTEST_3.npy")
liver_mask = np.load("D:\\MDS\\dataset\\Task03_Liver\\numpy\\val\\liverTEST_3.npy")

test_val_liver_input = np.expand_dims(test_liver, axis=0)
test_val_liver_prediction = my_model.predict(test_val_liver_input)
test_val_liver_prediction_argmax = np.argmax(test_val_liver_prediction, axis=4)[0, :, :, :]
test_mask_argmax = np.argmax(test_mask, axis=1)

n_slice_liver=random.randint(90,128)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_liver[:, :, n_slice_liver, 1], cmap='gray')
plt.subplot(232)
plt.title('Prediction on test image')
plt.imshow(test_val_liver_prediction_argmax[:, :, n_slice_liver])
plt.show()
