import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print tf.__version__
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()#dir:~/.keras/datasets
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
'''
plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i],cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128,activation=tf.nn.relu),
	keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=30)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print 'Test accuracy:',test_acc
predictions = model.predict(test_images)

def plot_image(i,predicts_array,true_labels,images):
	predictions,true_label,image = predicts_array[i],true_labels[i],images[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(image,cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel('{}{:2.0f}%({})'.format(class_names[predicted_label],
									   100*np.max(predictions),
									   class_names[true_label]),
									   color=color)

def plot_value_array(i,predictions_array,true_labels):
	predictions,true_label = predictions_array[i],true_labels[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10),predictions,color="#777777")
	plt.ylim([0,1])
	predicted_label = np.argmax(predictions)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, 2*i+1)
	plot_image(i, predictions, test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2*i+2)
	plot_value_array(i, predictions, test_labels) 
plt.show()
