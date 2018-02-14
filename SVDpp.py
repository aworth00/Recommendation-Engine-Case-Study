import time
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise import Dataset, evaluate


start_time = time.time()

data = Dataset.load_builtin('ml-1m')
e = 15
reg = .03
init_mean = .1
algo = SVDpp(verbose=1)
evaluate(algo, data)

running_time = time.time() - start_time
print("SVD:", running_time," s")
