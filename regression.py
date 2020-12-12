import numpy as np
import pandas as pd
import math

class lin_reg_model:
    def fit(self, X, Y, start_b=None, algorithm="RMSProp", reg_type='L1L2', reg_par=0):
        if start_b is None:
            start_b = np.zeros(len(X.columns) + 1)

        b = None
        err = None
        if algorithm == "RMSProp":
            self.b, self.err = RMSprop(X, Y, start_b, reg_type, reg_par, scale_coef=0.8, num_iter=50)
        elif algorithm == "Adam":
            self.b, self.err = Adam(X, Y, start_b, reg_type, reg_par, scale_coef_M=Y.mean(), scale_coef_G=0.1, reg_par=0.8)
        elif algorithm == "Adagrad":
            self.b, self.err = Adagrad(X, Y, start_b, reg_type, reg_par, num_iter=50)
        elif SGD:
            self.b, self.err = SGD(X, Y, start_b, reg_type, reg_par, num_iter=50, logging=False)

    def predict(self, X): return np.dot(X, self.b[:-1]) + self.b[-1]

    def get_weights(self): return self.b
    def get_err(self): return self.err

def prediction(x, b):
    return np.dot(np.append(x, 1), b)

def calculate_error(X, Y, b):
    pred = (X * b[:-1]).sum(axis=1) + b[-1]
    return ((Y - pred) ** 2).sum()

def calculate_error_with_regul(X, Y, b, regul_error):
    pred = (X * b[:-1] + b[-1]).sum(axis=1)
    return ((Y - pred) ** 2).sum() + regul_error

ERROR_REGUL_FUNC = {
    'L1': lambda X, Y, b, reg_par: calculate_error_with_regul(X, Y, b, (reg_par * abs(b[:-1])).sum()), # lasso
    'L2': lambda X, Y, b, reg_par: calculate_error_with_regul(X, Y, b, (reg_par * b[:-1] ** 2).sum()), # ridge
    'L1L2': lambda X, Y, b, reg_par: calculate_error_with_regul(X, Y, b, (reg_par * (b[:-1] ** 2 + abs(b[:-1]))).sum()),
}

def gradient(X,Y, b):
    h = 0.00001
    Error = calculate_error(X,Y, b)
    size = len(b)
    grad = np.zeros(size)
    for i in range(size):
        b[i] += h
        #grad[i] = (Error - calculate_error_by_Kfold(X,Y, b))/h
        grad[i] = (Error - calculate_error(X,Y, b))
        b[i] -= h
    return grad, Error

def gradient_with_regul(X, Y, b, reg_type, reg_par):
    h = 0.00001
    Error = ERROR_REGUL_FUNC[reg_type](X, Y, b, reg_par)
    size = len(b)
    grad = np.zeros(size)
    for i in range(size):
        b[i] += h
        grad[i] = (Error - ERROR_REGUL_FUNC[reg_type](X,Y, b, reg_par))
        b[i] -= h
    return grad, Error  

    
def SGD(X, Y, b, reg_type, reg_par, num_iter=50, logging=False):
    coef_grad = 1
    coef_prev_grad = 0.2
    prev_grad, prev_error = gradient_with_regul(X,Y,b, reg_type, reg_par)
    b = b + prev_grad * coef_grad
    error = prev_error + 1
    next_error = 0
    for c in range(1, num_iter + 1):
        if abs(error - prev_error) <= 0.001: break
        grad, error = gradient_with_regul(X,Y,b, reg_type, reg_par)
        b = b + grad * coef_grad + prev_grad * coef_prev_grad
        next_error = ERROR_REGUL_FUNC[reg_type](X,Y, b, reg_par)
        # проверка действительно ли следующий шаг норм и смена коеффициентов
        while next_error > error:
            coef_grad /= 2
            coef_prev_grad /= 2
            b = b - grad * coef_grad + prev_grad * coef_prev_grad
            next_error = ERROR_REGUL_FUNC[reg_type](X,Y, b, reg_par)
        else:
            coef_grad *= 2
            coef_prev_grad *= 2
            
        prev_grad = grad
        prev_error = error
        error = next_error
        if logging:
            print(f'{c}  :  {error=}')
        
    return b, calculate_error(X, Y, b)


def Adagrad(X, Y, b, reg_type, reg_par, num_iter = 50, logging=False):
    coef_grad = 0.1
    G_vector = np.zeros(len(b)) + 0.01
    G_vector_sqr = G_vector ** 2
    
    next_error = 0
    for c in range(1, num_iter + 1):
        grad, error = gradient_with_regul(X,Y,b, reg_type, reg_par)
        G_vector_sqr = G_vector_sqr + grad ** 2
        b = b + (grad * coef_grad) / G_vector
        
        next_error = ERROR_REGUL_FUNC[reg_type](X,Y, b, reg_par)
        # проверка действительно ли следующий шаг норм и смена коеффициентов
        while next_error > error:
            coef_grad /= 2
            b = b - (grad * coef_grad) / G_vector
            next_error = ERROR_REGUL_FUNC[reg_type](X,Y, b, reg_par)
        else:
            coef_grad *= 2
        G_vector = G_vector_sqr ** 0.5
        error = next_error
        if logging:
            print(f'{c}  :  {error=};  {coef_grad=}')
    return b, calculate_error(X, Y, b)


def RMSprop(X, Y, b, reg_type, reg_par, scale_coef=0.5, num_iter = 50, logging=False):
    coef_grad = 0.1
    G_vector = np.zeros(len(b)) + 0.01
    G_vector_sqr = G_vector ** 2

    next_error = 0
    for c in range(1, num_iter + 1):
        grad, error = gradient_with_regul(X, Y, b, reg_type, reg_par)
        
        G_vector_sqr = G_vector_sqr * scale_coef + (1 - scale_coef) * grad ** 2
        b = b + (grad * coef_grad) / G_vector
        
        next_error = ERROR_REGUL_FUNC[reg_type](X, Y, b, reg_par)
        # проверка действительно ли следующий шаг норм и смена коеффициентов
        while (next_error := ERROR_REGUL_FUNC[reg_type](X, Y, b, reg_par)) > error:
            coef_grad /= 2
            b = b - (grad * coef_grad) / G_vector
        else:
            coef_grad *= 2

        G_vector = G_vector_sqr ** 0.5
        error = next_error
        if logging:
            print(f'{c}  :  {error=};  {coef_grad=}')
    return b, calculate_error(X, Y, b)


def Adam(X, Y, b, reg_type, reg_par, scale_coef_M = 0.5, scale_coef_G = 0.5, num_iter = 50, logging=False):
    # здесь G_vector - то же самое что G_vector_sqr в прошлыых двух
    coef_grad = 0.1
    G_vector = np.zeros(len(b)) + 0.01
    M_vector = np.zeros(len(b))
    
    next_error = 0
    for c in range(1, num_iter + 1):
        grad, error = gradient_with_regul(X, Y, b, reg_type, reg_par)
        
        M_vector = M_vector * scale_coef_M + (1 - scale_coef_M) * grad
        G_vector = G_vector * scale_coef_G + (1 - scale_coef_G) * grad ** 2
        
        m_ = M_vector / (1 - scale_coef_M ** c)
        g_ = (G_vector / (1 - scale_coef_G ** c)) ** 0.5
        b = b + m_ * coef_grad / g_
        
        next_error = ERROR_REGUL_FUNC[reg_type](X,Y, b, reg_par)
        # проверка действительно ли следующий шаг норм и смена коеффициентов
        while next_error > error:
            coef_grad /= 2
            m_ = M_vector / (1 - scale_coef_M ** c)
            g_ = (G_vector / (1 - scale_coef_G ** c)) ** 0.5 
            b = b - m_ * coef_grad / g_
      
            next_error = ERROR_REGUL_FUNC[reg_type](X, Y, b, reg_par)
        else:
            coef_grad *= 2
        error = next_error
        if logging:
            print(f'{c}  :  {error=};  {coef_grad=}')
    return b, calculate_error(X, Y, b)


def k_fold_cross_validation(model, X, Y, k, fit_params={}):
    size_test = len(X) // k
    begin = 0
    end = size_test + len(X) % k
    accum_mse = 0
    accum_mae = 0
    for i in range(k):

        X_train = pd.concat([X[:begin], X[end:]])
        Y_train = pd.concat([Y[:begin], Y[end:]])
        X_test = X[begin:end]
        Y_test = Y[begin:end]

        model.fit(X_train, Y_train, **fit_params)
        Y_pred = model.predict(X_test)

        mse = ((Y_pred - Y_test) ** 2).sum() / len(Y_test)
        mae = (Y_pred - Y_test).abs().sum() / len(Y_test)
        accum_mse += mse
        accum_mae += mae
        print(f"Step {i+1}: MAE = {mae}, MSE = {mse}")

        begin += size_test
        end += size_test

    print(f"Average MAE: {accum_mae / k}")
    print(f"Average MSE: {accum_mse / k}")
