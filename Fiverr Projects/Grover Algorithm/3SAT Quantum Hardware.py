#Running Grover 3 SAT Problem on actual quantum Hardware
# Load IBM Quantum account
provider = IBMQ.load_account()

# Get the least busy backend
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                   not x.configuration().simulator and x.status().operational==True))

print("Least busy backend:", backend)

# Convert 3SAT instance to quantum circuit
qc = sat_to_circuit(input_3sat_instance)

# Transpile the circuit for the backend
transpiled_qc = transpile(qc, backend)

# Submit job to the backend
job = backend.run(transpiled_qc)

# Monitor the job
job_monitor(job)

# Get the result
result = job.result()
counts = result.get_counts()

# Display the result
print(counts)