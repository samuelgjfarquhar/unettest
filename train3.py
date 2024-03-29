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
from keras.models import load_model
from keras.metrics import MeanIoU

tf.config.run_functions_eagerly=True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.chdir('C:/Users/samue/BraTS2020/')

####################################################
train_img_dir = "D:/test/ttT/"
train_mask_dir = "D:/test/ttM/"

val_img_dir = "D:/test/npyVal/"
val_mask_dir = "D:/test/npyValMask/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
batch_size = 2

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

wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

steps_per_epoch = len(train_img_list) * 2
val_steps_per_epoch = len(val_img_list) * 2

model = unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=128,
                          IMG_CHANNELS=1,
                          num_classes=3)

model.compile(optimizer=optim, loss=focal_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)

# ##################################################################
# history = model.fit(train_img_datagen,
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=40,
#                     verbose=1,
#                     validation_data=val_img_datagen,
#                     validation_steps=val_steps_per_epoch,
#                     )

# os.chdir('C:/Users/samue/UNETliver/')
# model.save('liver.hdf5')
# os.chdir('C:/Users/samue/BraTS2020/')

# # plot the training and validation IoU and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# plt.plot(epochs, acc, 'y', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# #################################################

# my_model = load_model('liver.hdf5',
#                       compile=False)

# # Now all set to continue the training process.
# history2 = my_model.fit(train_img_datagen,
#                         steps_per_epoch=steps_per_epoch,
#                         epochs=1,
#                         verbose=1,
#                         validation_data=val_img_datagen,
#                         validation_steps=val_steps_per_epoch,
#                         )
# ########################################

my_model = load_model('liver.hdf5', 
                      custom_objects={
                          'dice_loss': dice_loss,
                          'focal_loss': focal_loss,
                          'dice_loss_plus_1focal_loss': total_loss,
                          'iou_score':sm.metrics.IOUScore(threshold=0.5)})

batch_size = 4  # Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list,
                               val_mask_dir, val_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

val_loss, val_dice = my_model.evaluate(val_img_datagen)
print(f"validation soft dice loss: {val_loss:.4f}")
print(f"validation dice coefficient: {val_dice:.4f}")

n_classes = 3
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


tester = np.load("D:\\test\\npyVal\\liver_42.npy")
testermask = np.load("D:\\test\\npyValMask\\liver_42.npy")

tt = np.expand_dims(tester, axis=0)
test_val_prediction = my_model.predict(tt)
test_val_prediction_argmax = np.argmax(test_val_prediction, axis=4)[0, :, :]

n_slice_val=100

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Validation Test Image')
plt.imshow(tester[:, :, n_slice_val, :], cmap='gray')
plt.subplot(232)
plt.title('Prediction on validation image')
plt.imshow(test_val_prediction_argmax[:, :, n_slice_val])
plt.subplot(233)
plt.title('mask')
plt.imshow(testermask[:, :, n_slice_val])
plt.show()

