import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    # pass
    mx = np.max(values)
    mn = np.min(values)
    if mx==mn:
        return np.full_like(values,c,dtype=float)

    return  c+((d-c)/(mx-mn))*(values-mn)
    
    