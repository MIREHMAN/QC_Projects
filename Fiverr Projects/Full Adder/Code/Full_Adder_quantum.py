from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, IBMQ

# Load IBM Quantum account
IBMQ.load_account()

class Circuit:
    def __init__(self):
        self.c = ClassicalRegister(2)
        self.q = QuantumRegister(5)
        self.qc = QuantumCircuit(self.q, self.c)

    def initialize_registers(self, a_init, b_init, cin_init):
        if a_init == 0:
            self.qc.initialize([1, 0], 0)  # Initialize 'a' to |0>
        else:
            self.qc.initialize([0, 1], 0)  # Initialize 'a' to |1>
    
        if b_init == 0:
            self.qc.initialize([1, 0], 1)  # Initialize 'b' to |0>
        else:
            self.qc.initialize([0, 1], 1)  # Initialize 'b' to |1>
    
        if cin_init == 0:
            self.qc.initialize([1, 0], 2)  # Initialize 'cin' to |0>
        else:
            self.qc.initialize([0, 1], 2)  # Initialize 'cin' to |1>

    def apply_sum_oracle(self):
        SUM = QuantumCircuit(4, name="SUM")
        SUM.cx(0, 3)
        SUM.cx(1, 3)
        SUM.cx(2, 3)
        self.qc.append(SUM.to_instruction(), [self.q[0], self.q[1], self.q[2], self.q[3]])
        self.qc.barrier()

    def apply_carry_oracle(self):
        CARRY = QuantumCircuit(5, name="CARRY")
        CARRY.ccx(0, 1, 4)
        CARRY.ccx(1, 2, 4)
        CARRY.ccx(0, 2, 4)
        self.qc.append(CARRY.to_instruction(), [self.q[0], self.q[1], self.q[2], self.q[3], self.q[4]])

    def measure_circuit(self):
        self.qc.measure(self.q[3], self.c[1])
        self.qc.measure(self.q[4], self.c[0])

    def simulate(self, backend):
        job = execute(self.qc, backend=backend, shots=1024)
        result = job.result()
        counts = result.get_counts(self.qc)
        return counts

# Choose the backend
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibm_brisbane')

# Instantiate and simulate the circuit for all initialization combinations
for a_init in range(2):
    for b_init in range(2):
        for cin_init in range(2):
            print(f"Initial states: a={a_init}, b={b_init}, cin={cin_init}")
            circuit = Circuit()
            circuit.initialize_registers(a_init, b_init, cin_init)
            circuit.apply_sum_oracle()
            circuit.apply_carry_oracle()
            circuit.measure_circuit()
            counts = circuit.simulate(backend)
            print(counts)
            print()
