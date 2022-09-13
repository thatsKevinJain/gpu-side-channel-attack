import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

FILE="large-data.csv"
K=5
TEST_SIZE=0.3
RUNS=100

data = pd.read_csv(FILE)
print(data.head())

le = preprocessing.LabelEncoder()

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
X = list(zip(used,sm_efficiency,achieved_occupancy,ipc,issued_ipc,issue_slot_utilization,tex_fu_utilization,stall_inst_fetch,stall_memory_dependency,stall_constant_memory_dependency,l2_read_transactions,l2_read_throughput,dram_utilization,eligible_warps_per_cycle,stall_exec_dependency,stall_other,stall_pipe_busy,stall_memory_throttle,stall_not_selected))

# labels
y = list(website)

sum = 0
for i in range(RUNS):
	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = TEST_SIZE)

	model = KNeighborsClassifier(n_neighbors=K)
	model.fit(x_train, y_train)

	acc = model.score(x_test, y_test)
	sum += (acc*100)
print()
print("Training Data:", FILE, "; K Neigbours:", K, "; Runs:", RUNS)
print()
print("Average Accuracy: ", sum/RUNS)

# predicted = model.predict(x_test)
# names = ["facebook", "google", "none"]

# for x in range(len(predicted)):
# 	print("Predicted: ", names[predicted[x]], "Actual: ", names[y_test[x]])

