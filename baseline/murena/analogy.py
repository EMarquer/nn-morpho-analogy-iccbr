import sys
try:
    from .utils import getTransformationPart1, getLengthInstruction, getPart2, applyTransformation, writeInstruction, getTransformation2
except ImportError:
    from utils import getTransformationPart1, getLengthInstruction, getPart2, applyTransformation, writeInstruction, getTransformation2


analogy_buffer = {}


def solveAnalogy(A, B, C):
    """
    Returns a solution (or a list of valid solutions) for 
    the analogical equation "`A` : `B` # `C` : x", where 
    each solution is constituted of the solution term D 
    and the corresponding transformation.
    """
    
    abc = A + ':' + B + '::' + C
    
    if abc in analogy_buffer: 
        #print('using analogy buffer')
        return analogy_buffer[abc]
    
    min_length_result = len(C) + len(B) - len(A)

    final_result = []

    result_transf_1 = []
    result_varA = []
    result_varC = []
    list_varA = []
    list_varC = []

    getTransformationPart1("", A, C, list_varA, list_varC, result_transf_1, result_varA, result_varC)
    min_length = sys.maxsize
    
    for x in range(len(result_transf_1)):
        ll = getLengthInstruction(result_transf_1[x], result_varA[x] + result_varC[x])
        if (ll <= min_length):
            result_transf_2 = []
            result_varB = []
            l = result_varA[x]
            getTransformation2(result_transf_1[x] + ",:", B, l, result_transf_2, result_varB)
            for y in range(len(result_transf_2)):

                ll = getLengthInstruction(result_transf_2[y], result_varB[y] + result_varC[x])
                if (ll <= min_length):
                    partInstruction_B = getPart2(result_transf_2[y])
                    result_varD = list(result_varC[x])
                    D = applyTransformation(partInstruction_B, result_varD)
                    ll = getLengthInstruction(result_transf_2[y], result_varB[y] + result_varD)

                    if (ll < min_length and len(D) >= min_length_result):
                        min_length = ll
                        final_result = [ [D, writeInstruction(result_transf_2[y], result_varB[y], result_varD)] ]
                    elif (ll == min_length and len(D) >= min_length_result):
                        final_result.append([D, writeInstruction(result_transf_2[y], result_varB[y], result_varD)])
                        
    analogy_buffer[abc] = (final_result, min_length)
    return final_result, min_length






def solveAnalogy_proba(A, B, C):
    """
    Returns a solution (or a list of valid solutions) for 
    the analogical equation "`A` : `B` # `C` : x", where 
    each solution is constituted of the solution term D 
    and the corresponding transformation.
    """
    
    possible_results = {}

    result_transf_1 = []
    result_varA = []
    result_varC = []
    list_varA = []
    list_varC = []

    getTransformationPart1("", A, C, list_varA, list_varC, result_transf_1, result_varA, result_varC)
    min_length = sys.maxsize
    
    for x in range(len(result_transf_1)):
        ll = getLengthInstruction(result_transf_1[x], result_varA[x] + result_varC[x])
        if (ll <= min_length):
            result_transf_2 = []
            result_varB = []
            l = result_varA[x]
            getTransformation2(result_transf_1[x] + ",:", B, l, result_transf_2, result_varB)
            for y in range(len(result_transf_2)):

                ll = getLengthInstruction(result_transf_2[y], result_varB[y] + result_varC[x])
                if (ll <= min_length):
                    partInstruction_B = getPart2(result_transf_2[y])
                    result_varD = list(result_varC[x])
                    D = applyTransformation(partInstruction_B, result_varD)
                    ll = getLengthInstruction(result_transf_2[y], result_varB[y] + result_varD)
                    
                    if D in possible_results:
                        possible_results[D] += 2**(-ll)
                    else: 
                        possible_results[D] = 2 **(-ll)
                        
    # Normalization
    factor = 1.0/sum(possible_results.values())
    for D in possible_results: possible_results[D] *= factor
    return sorted(possible_results.items(), key=lambda item: item[1], reverse=True)



def classification(A, B, C, D, n=1):
    if n == 1: 
        res = solveAnalogy(A,B,C)
        return D in [res[0][i][0] for i in range(len(res[0]))]
    res = solveAnalogy_proba(A,B,C)
    return D in [res[i][0] for i in range(n)]

# extra code

def classification_(A, B, C, D, n=1):
    if n == 1: 
        res = solveAnalogy(A,B,C)
        l = [res[0][i][0] for i in range(len(res[0]))]
    else:
        res = solveAnalogy_proba(A,B,C)
        l = [res[i][0] for i in range(min(n, len(res)))]
        
    if D in l:
        return True, l.index(D)
    else:
        return False, 99999999

def classification_with_results(A, B, C, D, n=1):
    if n == 1: 
        res = solveAnalogy(A,B,C)
        l = [res[0][i][0] for i in range(len(res[0]))]
    else:
        res = solveAnalogy_proba(A,B,C)
        l = [res[i][0] for i in range(min(n, len(res)))]
        
    if D in l:
        return True, l.index(D), l
    else:
        return False, 99999999, l