## Peter's integer solver!!!!

import cvxopt.glpk as glpk
import numpy as np
import cvxopt

def values(sol, data, problem=None, varnames=None):
    '''
    Returns aggregate scores given a solution.
    If a problem is provided, will include the overall objective score.
    '''
    # dvec = choiceArrayToBinary(sol, data)
    nparcels = data['nparcels']
    nopts = data['nopts']
    ndvs = nparcels * nopts
    results = dict()
    if varnames is None:
        factors = data['factornames']
    else:
        factors = varnames
    # calculate total scores for each desired variable
    for factor in factors:
        d = data[factor]
        results[factor] = sum([d[i][sol[i]] for i in range(nparcels)])
    # calculate total weighted objective
    if problem is not None:
        cv = [0 for _ in range(nparcels)]
        for i in range(nparcels):
            for j in factors:
                cv[i] = cv[i] + data[j][i][sol[i]]*problem['weights'][j]
        results['objective'] = sum(cv)
    return results


def optimize(data, problem, undefined_array=None,
             tiebreaker_intervention=None):
    '''
    data should be a dict structured as follows:
        'factornames': [ names of factors ]
        'factorname1': data for factor 1
        'factorname2': data for factor 2
        ...
        'choicenames': [name for each discretechoice] (optional)
        'choicemask': array with 0 indicating a conversion is prohibited (optional)

    problem should be a dict structured as follows:
        'weights': {factornames}: weight for each factor <- i.e. another dict
        'targets': {factornames}: target for each factor with a target
        'targettypes': {factornames}: '<', '>', or '=' (inequals will allow ==)

    It is assumed that weight < 0 indicates that we want to minimize that factor,
    and a weight > 0 indicates we want to maximize. Since ilp() minimizes by default,
    we have to multiply the weights by -1.


    ilp def:
    (status, x) = ilp(c, G, h, A, b, I, B)
    solves the mixed integer linear programming problem
        minimize    c'*x
        subject to  G*x <= h
                    A*x = b
                    x[I] are all integer
                    x[B] are all binary

        c,G,h,A,b must all be 'd' matrices, (the capitalized ones may be sparse)
        I and B must be sets of indices

    In the main formulation, the choice variables are binary - one variable
    for each combination of parcel and management. All the x corresponding
    to a given parcel are constrained to sum to 1 (that is, only one of the
    management options can be chosen). Note this assumes that baseline, or no
    change is included in the management options.
    '''
    if 'targets' in problem.keys():
        targetobjs = problem['targets'].keys()
        ntargets = len(targetobjs)
    else:
        targetobjs = []
        ntargets = 0

    factors = data['factornames']
    nfactors = len(factors)

    nparcels, nopts = np.array(data[factors[0]]).shape
    ndvs = nparcels * nopts
    data['nparcels'] = nparcels
    data['nopts'] = nopts

    # Construct objective function vector
    c_arr = [0 for _ in range(ndvs)]
    for i in range(ndvs):
        dr,dc = np.unravel_index(i, (nparcels, nopts))
        for j in factors:
            c_arr[i] = c_arr[i] + -1*data[j][dr][dc]*problem['weights'][j]

    # enforce tiebreaker
    for i in range(nparcels):
        choice_list = np.array(c_arr[(i*nopts):((i*nopts) + nopts)])
        indices = np.where(choice_list == choice_list.min())
        if tiebreaker_intervention in indices[0]:
            c_arr[(i*nopts) + tiebreaker_intervention] = \
                             c_arr[(i*nopts) + tiebreaker_intervention] - 0.001
    
    # Remove undefined pixels (pixels of lulc type for which the intervention is
    # not defined)
    if undefined_array:
        max_val = max(c_arr)
        new_val = max_val + 10000
        for i in range(ndvs):
            dr,dc = np.unravel_index(i, (nparcels, nopts))
            if undefined_array[dr][dc] == 1:
                c_arr[i] = new_val

    # Construct Inequalities
    ineqfactors = [f[0] for f in problem['targettypes'].items() if f[1] in ('<', '>')]
    ninequalities = len(ineqfactors)
    if ninequalities == 0:
        G_arr = np.zeros((ndvs, 1), dtype=float)
        h_arr = np.zeros(1, dtype=float)
    else:
        G_arr = np.zeros((ndvs, ninequalities), dtype=float)
        h_arr = np.zeros(ninequalities, dtype=float)
        for i in range(ninequalities):
            f = ineqfactors[i]
            if problem['targettypes'][f] == '<':
                m = 1.0
            else:
                m = -1.0
            h_arr[i] = m * problem['targets'][f]
            for j in range(ndvs):
                dr,dc = np.unravel_index(j, (nparcels, nopts))
                G_arr[j][i] = m * data[f][dr][dc]


    # Construct Equalities
    # note that we don't really want equalities for services since we can only
    # get integer quantities of the data and it might be impossible to hit
    # the target exactly. I'm leaving the code in for special cases only.
    eqfactors = [f[0] for f in problem['targettypes'].items() if f[1]=='=']
    nequalities = len(eqfactors) + nparcels
    A_arr = np.zeros((ndvs, nequalities), dtype=float)
    b_arr = np.zeros(nequalities, dtype=float)
    # first make 1 choice per parcel constraints
    for i in range(nparcels):
        b_arr[i] = 1
        for j in range(nopts):
            k = i*nopts
            A_arr[j+k,i] = 1
    # for i in range(nparcels,nequalities):
    for i in range(len(eqfactors)):
        ii = i+nparcels
        f = eqfactors[i]
        b_arr[ii] = problem['targets'][f]
        for j in range(ndvs):
            dr,dc = np.unravel_index(j, (nparcels, nopts))
            A_arr[j][ii]=data[f][dr][dc]


    # Construct variable definition sets - all binary
    I = set()
    B = set(range(ndvs))

    # Convert everything to a cvxopt matrix
    c = cvxopt.matrix(c_arr)
    del c_arr
    G = cvxopt.sparse((cvxopt.matrix(np.transpose(G_arr))))
    del G_arr
    h = cvxopt.matrix(h_arr)
    del h_arr
    A = cvxopt.sparse((cvxopt.matrix(np.transpose(A_arr))))
    del A_arr
    b = cvxopt.matrix(b_arr)
    del b_arr

    # Run the optimizer
    glpk.options['msg_lev']='GLP_MSG_ERR' # turns off non-error output to terminal
    if data.has_key('glpkopts'):
        for k,v in data['glpkopts'].items():
            glpk.options[k] = v
    (status,x) = glpk.ilp(c,G,h,A,b,I,B)
    if x is None:
        print "cost constraint: " + str(problem['targets']['Cost'])
        print "status: " + status
        return None
    else:
        return getChoiceArray(x, data)

def getChoiceArray(x, data):
    nparcels = data['nparcels']
    nopts = data['nopts']
    opt = [0 for _ in range(int(nparcels))]
    for f in range(nparcels):
        d = f*nopts
        opt[f] = np.argmax(x[d:d+nopts])
    return opt

def choiceArrayToBinary(ca, data):
    nparcels = data['nparcels']
    nopts = data['nopts']
    ndvs = nparcels * nopts
    x = np.zeros(ndvs, dtype=float)
    for i in range(nparcels):
        x[i*ndvs+ca[i]] = 1.0
    return x

def constructResultsForTable(frontier, data, headers=None):
    '''
    returns a nested list with rows for
    '''
    if headers is None:
        varnames = data['factornames']
    else:
        varnames = headers

    solutions = frontier['solutions']

    nfactors = len(varnames)
    nsolutions = len(solutions)
    nparcels = data['nparcels']


    results = [ [0.0 for _ in range(nfactors)] for _ in range(nsolutions)]
    for s in range(nsolutions):
        solscores = values(solutions[s], data, varnames=varnames)
        for f in range(nfactors):
            results[s][f] = solscores[varnames[f]]

    return results


def parcelScores(dv, factor, data, byArea=False):
    d = data[factor]
    if byArea:
        r = [ d[i][dv[i]]/data['areas'][i] for i in range(len(dv)) ]
    else:
        r = [d[i][dv[i]] for i in range(len(dv))]
    return r
