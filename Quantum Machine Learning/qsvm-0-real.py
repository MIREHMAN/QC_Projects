from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# this is needed to get real hardware to run the job
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

# load the IRIS Data Set
from sklearn import datasets
iris = datasets.load_iris()

# this is the problematic area when using real hardware instead of the simulator
adhoc_dimension = len(iris.data[0])
adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
sampler = Sampler(backend=backend)
fidelity = ComputeUncompute(sampler=sampler)
adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

# IBM apparantly deprecated a concept called "abstract circuits" recently - this approach no longer works
#
# qiskit_ibm_runtime.exceptions.IBMInputValueError: 'The instruction ZZFeatureMap on qubits (0, 1, 2, 3) 
# is not supported by the target system. Circuits that do not match the target hardware definition are no 
# longer supported after March 4, 2024. See the transpilation documentation (https://docs.quantum.ibm.com/transpile) 
# for instructions to transform circuits and the primitive examples (https://docs.quantum.ibm.com/run/primitives-examples) 
# to see this coupled with operator transformations.'
#

qsvc = QSVC(quantum_kernel=adhoc_kernel)

qsvc.fit(iris.data,iris.target)

labels = iris.target_names
print(labels)

flowers = [
            [5.0, 3.3, 1.5, 0.3], # first unknown flower measurements
            [6.0, 2.9, 5.2, 1.7]  # second unknown flower measurements
          ]

predictions = qsvc.predict(flowers)

# this would print the predictions from the model if this worked
print("predictions:")
for i, p in enumerate(predictions):
    print("{} => {}".format(flowers[i], labels[p]))
