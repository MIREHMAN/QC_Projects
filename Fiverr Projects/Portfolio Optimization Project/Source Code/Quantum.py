from qiskit_algorithms.utils import algorithm_globals
from  qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, SamplingVQE
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler


#VQE Algorithm Application
algorithm_globals.random_seed = 1234
cobyla = COBYLA()
cobyla.set_options(maxiter=500)
ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
svqe = MinimumEigenOptimizer(svqe_mes)
result = svqe.solve(qp)

#Get Results using same print_result function
print_result(result)