import time
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import Dataset, evaluate, GridSearch

def run_svd(n_epochs, reg_all, init_mean):
    start_time = time.time()
    algo = SVD(n_epochs=n_epochs,reg_all=reg_all, init_mean=init_mean)
    evaluate(algo, data)
    running_time = time.time() - start_time
    print("SVD:", running_time," s")

def grid_search(param_grid):
    grid_search = GridSearch(SVD,param_grid,measures = ['RMSE'])
    grid_search.evaluate(data)
    print('Best RMSE',grid_search.best_score['RMSE'])
    return grid_search.best_params['RMSE']

if __name__ == "__main__":
    data = Dataset.load_builtin('ml-10m')
    e = 25
    reg = .03
    init_mean = .1
    algo = SVD()
    param_grid = {'n_epochs':e, 'reg_all':reg, 'init_mean':init_mean}
    # best_params = grid_search(param_grid)
    run_svd(e,reg,init_mean)
