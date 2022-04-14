import math


# tmp
max_nb_var = 3
# Length (in bits) of each word from the language
nb_char = 26 # /!\
length_letter = 3 + math.ceil(math.log(nb_char, 2)) # dynamically incremented for each analogical equation, based on the size of the alphabet
length_var = 4
length_gr = 2
length_let = 3
length_mem = 4
length_sep = 3
length_doublesep = 3
#--------------

def getLengthLetter():
    return length_letter


def getSubString(word):
    """
    Returns the prefixes of a word passed in as `w`.
    """
    
    substrings = []
    for i in reversed(range(len(word))):
        substrings.append(word[0:i+1])
    return substrings

def getLengthInstruction(transformation, list_var, nb_mem=2, nb_sep=1, nb_doublesep=1):
    """
    Returns the length, in bits, of the words included 
    in the transformation `transformation` and the words 
    considered as variables' instanciations and included 
    in `list_var`.
    """
    
    length = 0

    l_transf = transformation.split(",")
    for l in l_transf:
        if (l[0] == "'"):
            length += length_letter
        elif (l[0] == "?"):
            length += length_var + int(l[1:])

    for v in list_var:
        length += len(v) * length_letter

        if (len(v) > 1):
            length += 2 * length_gr
            
    length += 2 * length_let + nb_mem * length_mem + nb_sep * length_sep + nb_doublesep * length_doublesep
    return length

def getPart2(transformation):
    """
    Returns "part2" for every transformation of the form
    "part1,:,part2".
    """
    
    L = transformation.split(":")
    return L[1][1:]

def writeInstruction(transformation, list_var1, list_var2):
    """
    Formatting method, for readability purposes.
    For every tuple (transformation, list_var1, list_var2), 
    returns "let, `transformation`, let, 
             mem,0, `list_var1[0]`, `list_var1[1], ..., 
             #, mem,0, `list_var2[0]`, `list_var2[1]`, ...".    
    """
    
    s = "let," + transformation + ",let,mem,0"
    for el in list_var1:
        s += ",'" + el + "'"
    s += ",#,mem,0"
    for el in list_var2:
        s += ",'" + el + "'"   
    return s

def getTransformationPart1(transformation, A, C, list_varA, list_varC, result_transf, result_varA, result_varC):
    """
    Returns, in `result_transf`, a set of candidates for 
    the first part of the transformation, describing both 
    the terms `A` and `C`, along with the corresponding 
    variables' instanciations included in `result_varA` 
    and `result_varC`.
    """
    
    if ( (A == "" and C != "") or (A != "" and C == "")):
        return
    
    if (A == "" and C == ""):
        result_transf.append(transformation)
        result_varA.append(list_varA)
        result_varC.append(list_varC)
        
    elif transformation == "":
        #add letter
        if (A[0] == C[0]):
            _transformation = "'" + A[0] + "'"
            _A = A[1:]
            _C = C[1:]
            getTransformationPart1(_transformation, _A, _C, list_varA, list_varC, result_transf, result_varA, result_varC)
        
        #add first variable
        _transformation = "?0"
        for s_A in getSubString(A):
            for s_C in getSubString(C):
                getTransformationPart1(_transformation, A[len(s_A):], C[len(s_C):], [s_A], [s_C], result_transf, result_varA, result_varC)
        
    else:
        #add letter
        if (A[0] == C[0]):
            _transformation = transformation + ",'" + A[0] + "'"
            _A = A[1:]
            _C = C[1:]
            getTransformationPart1(_transformation, _A, _C, list_varA, list_varC, result_transf, result_varA, result_varC)       
             
        nb_var = len(list_varA)
           
        #add new variable
        if (nb_var < max_nb_var):
            _transformation = transformation + ",?" + str(nb_var)        
            for s_A in getSubString(A):
                for s_C in getSubString(C):
                    if (nb_var == 0):
                        getTransformationPart1(_transformation, A[len(s_A):], C[len(s_C):], [s_A], [s_C], result_transf, result_varA, result_varC)
                    else:
                        l_A = list(list_varA)
                        l_C = list(list_varC)
                        l_A.append(s_A)
                        l_C.append(s_C)
                        getTransformationPart1(_transformation, A[len(s_A):], C[len(s_C):], l_A, l_C, result_transf, result_varA, result_varC)
                
        #add existing variable
        for v in range(nb_var):
            if ((A[:len(list_varA[v])] == list_varA[v]) and (C[:len(list_varC[v])] == list_varC[v])):
                _transformation = transformation + ",?" + str(v)
                getTransformationPart1(_transformation, A[len(list_varA[v]):], C[len(list_varC[v]):], list_varA, list_varC, result_transf, result_varA, result_varC)


def getTransformation2(transformation, B, list_var, result_transf, result_var):
    """
    Returns, in `result_transf`, a set of candidates for 
    the transformation, by appending to the first part 
    of the transformation a second part that describes
    the term `B`, along with the corresponding variables' 
    instanciations included in `result_var`. 
    """
    
    if B == "":
        result_transf.append(transformation)
        result_var.append(list_var)
        
    elif transformation == "":
        #add letter
        _transformation = "'" + B[0] + "'"
        _B = B[1:]
        getTransformation2(_transformation, _B, list_var, result_transf, result_var)
        
        #add first variable
        _transformation = "?0"
        for s in getSubString(B):
            getTransformation2(_transformation, B[len(s):], [s], result_transf, result_var)
        
    else:
        nb_var = len(list_var)
        
        #add existing variable
        for v in range(nb_var):
            if (B[:len(list_var[v])] == list_var[v]):
                _transformation = transformation + ",?" + str(v)
                getTransformation2(_transformation, B[len(list_var[v]):], list_var, result_transf, result_var)
                     
        #add letter
        _transformation = transformation + ",'" + B[0] + "'"
        _B = B[1:]
        getTransformation2(_transformation, _B, list_var, result_transf, result_var)       
                  

def applyTransformation(transformation2, list_var):
    """
    Returns the solution D, given the second part of the 
    transformation `transformation2` and the corresponding 
    variables' instanciations included in `list_var`.
    """
    
    L = transformation2.split(",")
    D = ""
    for x in range(0, len(L)):
        if (L[x][0] == "'"):
            D += L[x][1]
        elif (L[x][0] == "?"):
            i = int(L[x][1:])
            if (i < len(list_var)):
                D += list_var[i]
            else:
                D += "*"
                list_var.append('*')
    return D 