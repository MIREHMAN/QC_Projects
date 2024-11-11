from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.compiler import transpile

# this is needed to get real hardware to run the job
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

# load the IRIS Data Set
from sklearn import datasets
iris = datasets.load_iris()

# as a debugging step for the real hardware, i had the idea to simplify the problem by only considering half of the training data columns
shortData = []

for sl in iris.data:
    shortData.append([sl[0], sl[3]])

# and only taking half of the rows
shortData = shortData[1::2]
shortTarget = iris.target[1::2]

print(shortData)
print(iris.target)

# this is the problematic area when using real hardware instead of the simulator
adhoc_dimension = len(shortData[0])
adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
sampler = Sampler(backend=backend)
fidelity = ComputeUncompute(sampler=sampler)

# transpiling due to IBM's deprecation of "abstract circuits" like ZZFeatureMap on real hardware
tp = transpile(adhoc_feature_map, backend=backend)
adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=tp)

# the transpile completes fine locally, however upon submitting the job...

qsvc = QSVC(quantum_kernel=adhoc_kernel)

qsvc.fit(shortData,shortTarget)

# you still get an error becuse ComputeUncompute introduces a low-level operation not supported by the real hardware
#
# qiskit_ibm_runtime.exceptions.IBMInputValueError: 'The instruction sxdg on qubits (1,) is not supported by the target system. 
# Circuits that do not match the target hardware definition are no longer supported after March 4, 2024. See the transpilation 
# documentation (https://docs.quantum.ibm.com/transpile) for instructions to transform circuits and the primitive examples 
# (https://docs.quantum.ibm.com/run/primitives-examples) to see this coupled with operator transformations.'
#

labels = iris.target_names
print(labels)

flowers = [
            [5.0, 1.5], # first unknown flower measurements
            [6.0, 5.2]  # second unknown flower measurements
          ]

predictions = qsvc.predict(flowers)

# this would print the (garbage) predictions from the model if this worked
print("predictions:")
for i, p in enumerate(predictions):
    print("{} => {}".format(flowers[i], labels[p]))
