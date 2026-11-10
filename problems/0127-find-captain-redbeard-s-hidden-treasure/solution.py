import scipy.optimize as opt
def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """
    # Your code here
    # pass
    # x = start_x
    # lr = 0.001
    # t = 1e-6
    # mi = 10000
    # for _ in range(mi):
    #   g = 4*(x**3)-9*(x**2)
    #   nx = x - lr*g
    #   if abs(nx-x)<t:
    #     return round(nx,4)
    #   x = nx
    
    # return round(x,4)
    def f(x):
      return (x**4)-3*(x**3)+2
    r = opt.minimize(f,x0= [start_x],method = 'Nelder-Mead')
    return float(r.x[0])