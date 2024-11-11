#Code for Simulation

from qiskit import Aer, QuantumCircuit, transpile, assemble,IBMQ
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
import numpy as np

# Define the 3SAT instance
input_3sat_instance = '''
c example DIMACS-CNF 3-SAT
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 -2 3 0
'''

# Function to convert 3SAT instance to quantum circuit
def sat_to_circuit(instance):
    clauses = instance.strip().split('\n')[2:]
    num_qubits = 3  # Number of qubits needed for this problem

    # Create a quantum circuit with num_qubits qubits
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Encode each clause into quantum gates
    for clause in clauses:
        literals = list(map(int, clause.split()[:-1]))
        qubit_indices = [abs(literal) - 1 for literal in literals]
        
        # Apply the OR gate for each clause
        qc.or_gate(qubit_indices)

    # Measure the qubits
    qc.measure(range(num_qubits), range(num_qubits))

    return qc

# Custom function to implement OR gate
def or_gate(self, qubit_indices):
    for index in qubit_indices[:-1]:
        self.cx(index, qubit_indices[-1])  # Apply CX (CNOT) gate
    return self

# Add custom OR gate function to QuantumCircuit class
QuantumCircuit.or_gate = or_gate

# Convert 3SAT instance to quantum circuit
qc = sat_to_circuit(input_3sat_instance)

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
transpiled_qc = transpile(qc, simulator)
qobj = assemble(transpiled_qc,shots=1000)
result = simulator.run(qobj).result()
counts = result.get_counts()

# Display the result
print(counts)

