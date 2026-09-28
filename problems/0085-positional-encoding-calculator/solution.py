import numpy as np

def pos_encoding(position: int, d_model: int):
    # 1. Compute positions and dimensions using standard float64 precision
    pos = np.arange(position)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    
    # 2. Safely calculate the angle frequencies
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates
    
    # 3. Apply sine and cosine in standard precision
    sines_cosines = np.zeros((position, d_model))
    sines_cosines[:, 0::2] = np.sin(angle_rads[:, 0::2])
    sines_cosines[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # 4. Cast the final output matrix to float16 to match your exact precision target
    return sines_cosines.astype(np.float16)

# print(pos_encoding(2, 8))
