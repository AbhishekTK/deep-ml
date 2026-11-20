import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
    # Your code here
    # passreturn 
    return [0.5*xi*(1+np.tanh(np.sqrt(2/np.pi)*(xi+0.044715*(xi**3)))) for xi in x]
    