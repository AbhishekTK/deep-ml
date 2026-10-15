
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    # pass
    minx = min(x)
    maxx = max(x)
    r = []
    for e in x:
        r.append((e-minx)/(maxx-minx))
    return r