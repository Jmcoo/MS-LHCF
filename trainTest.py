import numpy as np
import os
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

## set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_image_path = 'datasets/IHC/training set/'
test_image_path = 'datasets/IHC/testing set/'

train_image_list = []


def get_data(Dir):
    image_list = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            temp = Dir + nextDir

            for file in tqdm(os.listdir(temp)):
                image = temp + '/' + file
                image_list.append(image)

    random.shuffle(image_list)

    return image_list


# def get_data(Dir):
#     N_data = 0
#     for nextDir in os.listdir(Dir):
#         if not nextDir.startswith('.'):
#             temp = Dir + nextDir
#             data_path = os.listdir(temp)
#             N_data = N_data + len(data_path)
#     return N_data


N_train = get_data(train_image_path)
N_test = get_data(test_image_path)


def data_generator(image_list, batch_size=32):
    X = []
    y = []
    while True:
        for file in image_list:
            dir = file.split('/')[3]
            if dir in ['NNL']:
                label = 0
            elif dir in ['Cirrhosis']:
                label = 1
            elif dir in ['HCA']:
                label = 2
            elif dir in ['HGDN']:
                label = 3
            elif dir in ['HCC']:
                label = 4
            image = load_img(file, target_size=(256, 256))
            image = img_to_array(image)
            image /= 255
            X.append(image)
            y.append(label)
            if len(X) == batch_size:
                yield np.asarray(X), to_categorical(np.asarray(y), 5)
                X = []
                y = []


x_train_generator = data_generator(N_train)
x_test_generator = data_generator(N_test)

model = ResNet50(include_top=True,
                 weights=None,
                 input_tensor=None,
                 # input_shape=(256, 256, 3),
                 pooling=None)

model.load_weights('weights/weights_res/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
# define a new output layer to connect with the last fc layer
x = model.layers[-2].output
output_layer = Dense(5, activation='softmax')(x)

# combine the original model with the new output layer
model2 = Model(inputs=model.input, outputs=output_layer)

# compile the new model
model2.compile(optimizer=Adam(learning_rate=0.00002),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

filepath = 'weights/weights_res/res_ihc_256.h5'

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

hist = model2.fit(x_train_generator,
                  steps_per_epoch=len(N_train) // 32,
                  epochs=100,
                  verbose=1,
                  callbacks=[checkpoint],
                  validation_data=x_test_generator,
                  validation_steps=len(N_test) // 32,
                  shuffle=True)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig('res_ihc_256_loss.png', bbox_inches='tight')
plt.close()
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig('res_ihc_256_accuracy.png', bbox_inches='tight')
