from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def load_image(img):
    img=load_img(img,target_size=(64,64))
    img=img_to_array(img)
    img=img.reshape(1,64,64,3)
    img=img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

classifier=Sequential()

'''
Below I have added the Convolution and Maxpooling layers.
(32,3,3) means that 32 feature detectors of 3X3 each are to be generated. 
(64,64,3) indicates the resolution of the image with 3(RGB) or 2(Grayscale/Binary).

'''
classifier.add(Convolution2D(32,3,3,activation='relu',input_shape=(64,64,3)))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

'''
Taken the output dimension as 128 because we can't determine the number of nodes 
at the input layer. Hence tried 128. This can be changed for trial and error. 
'''

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#optimizer has the responsibility of updating thw weights to minimize the loss function
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

'''
->Image augmentation: Operations like Rotation, Shearing, Zooming (in/out), etc. are performed on the image to prevent overfitting. 
  Many batches are created of the images and each batch will create new images based on the above operations so that we have have more 
  number of images to be trained.
->One of the cases that leads to overfitting is when the model can successfully detect patterns and correlations
  between the observations present in the dataset but fails to detect them in the new observations that are provided as an input to the model. 
'''



#rescale=1./255 converts the pixel values from 0-255 to 0-1.
train_data=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_data=ImageDataGenerator(rescale=1./255)

train_set=train_data.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')

test_set=test_data.flow_from_directory('dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')


#samples_per_epoch indicates the number of samples we're passing as input from the training set.
#nb_val_samples indicates the number of samples we're using for testing present in the test set. .
classifier.fit_generator(train_set,samples_per_epoch=8000,nb_epoch=25,validation_data=test_set,nb_val_samples=2000)

#saving the features in a h5 file.
classifier.save('saved_feat.h5')

img=load_image('cat.jpg')

model=load_model('saved_feat.h5')

result=model.predict(img)

print(result[0])