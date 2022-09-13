# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

FILE="train.csv"
TEST_SIZE=0.3

data = pd.read_csv(FILE)
print(data.head())

le = preprocessing.LabelEncoder()

# Encode all columns #
used = le.fit_transform(list(data["used"]))
sm_efficiency = le.fit_transform(list(data["sm_efficiency"]))
achieved_occupancy = le.fit_transform(list(data["achieved_occupancy"]))
ipc = le.fit_transform(list(data["ipc"]))
issued_ipc = le.fit_transform(list(data["issued_ipc"]))
issue_slot_utilization = le.fit_transform(list(data["issue_slot_utilization"]))
tex_fu_utilization = le.fit_transform(list(data["tex_fu_utilization"]))
stall_inst_fetch = le.fit_transform(list(data["stall_inst_fetch"]))
stall_memory_dependency = le.fit_transform(list(data["stall_memory_dependency"]))
stall_constant_memory_dependency = le.fit_transform(list(data["stall_constant_memory_dependency"]))
l2_read_transactions = le.fit_transform(list(data["l2_read_transactions"]))
l2_read_throughput = le.fit_transform(list(data["l2_read_throughput"]))
dram_utilization = le.fit_transform(list(data["dram_utilization"]))
eligible_warps_per_cycle = le.fit_transform(list(data["eligible_warps_per_cycle"]))
stall_exec_dependency  = le.fit_transform(list(data["stall_exec_dependency"]))
stall_other  = le.fit_transform(list(data["stall_other"]))
stall_pipe_busy  = le.fit_transform(list(data["stall_pipe_busy"]))
stall_memory_throttle  = le.fit_transform(list(data["stall_memory_throttle"]))
stall_not_selected  = le.fit_transform(list(data["stall_not_selected"]))
website = le.fit_transform(list(data["website"]))

# features
X = np.array(list(zip(used,sm_efficiency,achieved_occupancy,ipc,issued_ipc,issue_slot_utilization,tex_fu_utilization,stall_inst_fetch,stall_memory_dependency,stall_constant_memory_dependency,l2_read_transactions,l2_read_throughput,dram_utilization,eligible_warps_per_cycle,stall_exec_dependency,stall_other,stall_pipe_busy,stall_memory_throttle,stall_not_selected)))

# labels
y = np.array(list(website))

# Split Datasets #
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = TEST_SIZE)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# MODEL #
model = tf.keras.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=4))
model.add(layers.LSTM(64))
model.add(layers.Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1)

# TEST ACCURACY #
test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)


## Kaggle Top Performed Models for similar Datasets
## Mitigation Techniques to avoid side channel attacks 







