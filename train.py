# Valohai workshop
import os
import json
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	'--epochs',
	type=int,
	default=5,
)
parser.add_argument(
	'--learning_rate',
	type=float,
	default=0.001,
)

params = parser.parse_args()

print('hello synergia')

# Get data which is given to valohai yaml config
input_path = os.getenv('VH_INPUTS_DIR','/valohai/inputs')
data = os.path.join(input_path, 'mnist/mnist.npz')

with np.load(data, allow_pickle=True) as f:
	x_train, y_train = f['x_train'], f['y_train']
	x_test, y_test = f['x_test'], f['y_test']

	
# Mnist dataset, handwritten digits
#mnist = tf.keras.datasets.mnist

# Refactor data -> teast and train datasets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define model -> network layers and nodes
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callback functions to enable valohai metrics
def log(epoch, logs):
	# print empty for valohai bug
	print()
	# dictionary comprehension, rewrite dictionary to make logs json serializable
	logs = {k: str(v) for(k,v) in logs.items()}
	print(json.dumps(
		{
			'epoch':epoch,
			'acc': str(logs['acc']),
			'loss': str(logs['loss']),
		}
	))
	
callback = tf.keras.callbacks.LambdaCallback(
	on_epoch_end=log
)
	
model.fit(x_train, y_train, epochs=params.epochs, callbacks=[callback])

path = os.getenv('VH_OUTPUTS_DIR', '/valohai/outputs')
model.save(os.path.join(path, 'model.h5'))

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(json.dumps(
	{
	'test_loss':str(test_loss),
	'test_acc':str(test_acc),
	}
))

print("\nSUMMARY\n")
print(model.summary())