from qiskit import *
import matplotlib.pyplot as plt
import numpy as np

oracle=QuantumCircuit(3,name='oracle')
oracle.cz(1,2)
oracle.to_gate()
oracle.draw()

backend=Aer.get_backend('statevector_simulator')
groover_cirq=QuantumCircuit(3,3)
groover_cirq.h([0,1,2])
groover_cirq.append(oracle,[0,1,2])
groover_cirq.draw()

jobs=execute(groover_cirq,backend)
result=jobs.result()
sv=result.get_statevector()
np.around(sv,2)

reflection=QuantumCircuit(3,name='reflection')
reflection.h([0,1,2])
reflection.z([0,1,2])
reflection.cz(0,1)
reflection.cz(0,2)
reflection.cz(1,2)
reflection.h([0,1,2])
reflection.to_gate()

backend=Aer.get_backend('qasm_simulator')
groover_circ=QuantumCircuit(3,3)

groover_circ.h([0,1,2])
groover_circ.append(oracle,[0,1,2])
groover_circ.append(reflection,[0,1,2])
groover_circ.measure([0,1,2],[0,1,2])

job=execute(groover_circ,backend,shots=1)
result=job.result()
result.get_counts()