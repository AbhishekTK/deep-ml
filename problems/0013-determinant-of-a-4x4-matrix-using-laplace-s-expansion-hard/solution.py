def helper_determinant_3X3(m: list[list[int|float]]) -> float:
    cf = 0
    i = 0
    for j in range(len(m[0])):
        cf += ((-1)**(i+j)) * (m[i][j]) * extract_cofactor_matrix_2X2(m,i,j)
        # print("3X3 cf",cf)
    return cf
def extract_sub_matrix(m:list[list[int|float]],r:int, c:int) -> list[list[int|float]]:
    # res = [ [] for i in range(2) ]
    res = []
    k = 0
    for i in range(len(m)):
        row = []
        if i == r:
            continue
        for j in range(len(m[0])):
            if j == c:
                continue
            row.append(m[i][j])
        res.append(row)
    return res
def extract_cofactor_matrix_2X2(m:list[list[int|float]],r:int, c:int) -> int|float:
    # res = [ [] for i in range(2) ]
    res = []
    k = 0
    for i in range(len(m)):
        row = []
        if i == r:
            continue
        for j in range(len(m[0])):
            if j == c:
                continue
            row.append(m[i][j])
  