import numpy as np
from scipy.stats import levy_stable

def init(search_agents, dim, ub, lb):
    ub = np.array(ub)
    lb = np.array(lb)

    return np.random.rand(search_agents, dim) * (ub - lb) + lb

def mpa(search_agents, max_iter, lb, ub, dim, fobj):
    top_predator_pos = np.zeros((1, dim))
    top_predator_fit = np.inf

    convergence_curve = np.zeros((1, max_iter))
    step = np.zeros((search_agents, dim))
    fitness = np.full_like(np.ndarray((search_agents, 1)), np.inf)

    prey = init(search_agents, dim, ub, lb)

    xmin = np.tile(np.ones((1, dim)) * lb, (search_agents, 1))
    xmax = np.tile(np.ones((1, dim)) * ub, (search_agents, 1))

    iter = 0
    fads = 0.2
    p = 0.5

    while iter < max_iter:
        # detecting top predator
        for i in range(prey.shape[0]):
            prey[i, :] = np.clip(prey[i, :], lb, ub)
            fitness[i, 0] = fobj(prey[i, :])

            if fitness[i, 0] < top_predator_fit:
                top_predator_fit = fitness[i, 0]
                top_predator_pos = prey[i, :]

        # marine memory saving
        if iter == 0:
            fit_old = fitness
            prey_old = prey

        inx = (fit_old < fitness)
        indx = np.tile(inx, (1, dim))
        prey[indx] = prey_old[indx]
        fitness[inx] = fit_old[inx]

        fit_old = fitness
        prey_old = prey

        #
        elite = np.tile(top_predator_pos, (search_agents, 1))
        cf = (1 - iter/max_iter) ** (2 * iter/max_iter)

        rl = 0.05 * levy_stable.rvs(1.5, 0, size=(search_agents, dim))
        rb = np.random.randn(search_agents, dim)

        for i in range(prey.shape[0]):
            for j in range(prey.shape[1]):
                r = np.random.rand()

                if iter < max_iter/3:
                    step[i, j] = rb[i, j] * (elite[i, j] - rb[i, j] * prey[i, j])
                    prey[i, j] = prey[i, j] + p * r * step[i, j]

                elif iter > max_iter/3 and iter < 2 * max_iter/3:
                    if i > prey.shape[0]/2:
                        step[i, j] = rb[i, j] * (rb[i, j] * elite[i, j] - prey[i, j])
                        prey[i, j] = elite[i, j] + p * cf * step[i, j]
                    else:
                        step[i, j] = rl[i, j] * (elite[i, j] - rl[i, j] * prey[i, j])                    
                        prey[i, j] = prey[i, j] + p * r * step[i, j]
                else:
                    step[i, j] = rl[i, j] * (rl[i, j] * elite[i, j] - prey[i, j])
                    prey[i, j] = elite[i, j] + p * cf * step[i, j]

            # detecting top predator (again) (can make this a function, maybe)
            for i in range(prey.shape[0]):
                prey[i, :] = np.clip(prey[i, :], lb, ub)
                fitness[i, 0] = fobj(prey[i, :])

                if fitness[i, 0] < top_predator_fit:
                    top_predator_fit = fitness[i, 0]
                    top_predator_pos = prey[i, :]

        # marine memory saving again also
        if iter == 0:
            fit_old = fitness
            prey_old = prey

        inx = (fit_old < fitness)
        indx = np.tile(inx, (1, dim))
        prey[indx] = prey_old[indx]
        fitness[inx] = fit_old[inx]

        fit_old = fitness
        prey_old = prey

        # eddy formation and FADs effect
        if np.random.rand() < fads:
            u = np.random.rand(search_agents, dim) < fads
            prey = prey + cf * ((xmin + np.random.rand(search_agents, dim) * (xmax - xmin) * u))
        else:
            r = np.random.rand()
            rs = prey.shape[0]
            step = (fads * (1 - r) + r) * (prey[np.random.permutation(rs), :] - prey[np.random.permutation(rs), :])
            prey = prey + step
        iter += 1
        convergence_curve[0, iter - 1] = top_predator_fit
    
    return top_predator_fit, top_predator_pos, convergence_curve

def Ufun(x, a, k, m):
    x = np.array(x)
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)

def get_function_details(F):
    functions = {
        "F1":  {"fobj": F1,  "lb": -100,     "ub": 100,     "dim": 50},
        "F2":  {"fobj": F2,  "lb": -10,      "ub": 10,      "dim": 50},
        "F3":  {"fobj": F3,  "lb": -100,     "ub": 100,     "dim": 50},
        "F4":  {"fobj": F4,  "lb": -100,     "ub": 100,     "dim": 50},
        "F5":  {"fobj": F5,  "lb": -30,      "ub": 30,      "dim": 50},
        "F6":  {"fobj": F6,  "lb": -100,     "ub": 100,     "dim": 50},
        "F7":  {"fobj": F7,  "lb": -1.28,    "ub": 1.28,    "dim": 50},
        "F8":  {"fobj": F8,  "lb": -500,     "ub": 500,     "dim": 50},
        "F9":  {"fobj": F9,  "lb": -5.12,    "ub": 5.12,    "dim": 50},
        "F10": {"fobj": F10, "lb": -32,      "ub": 32,      "dim": 50},
        "F11": {"fobj": F11, "lb": -600,     "ub": 600,     "dim": 50},
        "F12": {"fobj": F12, "lb": -50,      "ub": 50,      "dim": 50},
        "F13": {"fobj": F13, "lb": -50,      "ub": 50,      "dim": 50},
        "F14": {"fobj": F14, "lb": -65.536,  "ub": 65.536,  "dim": 2},
        "F15": {"fobj": F15, "lb": -5,       "ub": 5,       "dim": 4},
        "F16": {"fobj": F16, "lb": -5,       "ub": 5,       "dim": 2},
        "F17": {"fobj": F17, "lb": [-5, 0],  "ub": [10,15], "dim": 2},
        "F18": {"fobj": F18, "lb": -2,       "ub": 2,       "dim": 2},
        "F19": {"fobj": F19, "lb": 0,        "ub": 1,       "dim": 3},
        "F20": {"fobj": F20, "lb": 0,        "ub": 1,       "dim": 6},
        "F21": {"fobj": F21, "lb": 0,        "ub": 10,      "dim": 4},
        "F22": {"fobj": F22, "lb": 0,        "ub": 10,      "dim": 4},
        "F23": {"fobj": F23, "lb": 0,        "ub": 10,      "dim": 4},
    }

    config = functions[F]
    lb = np.array(config["lb"])
    ub = np.array(config["ub"])

    return lb, ub, config["dim"], config["fobj"]

def F1(x):
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    x = np.array(x)
    o = 0
    for i in range(len(x)):
        o += np.sum(x[:i+1])**2
    return o

def F4(x):
    return np.max(np.abs(x))

def F5(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def F6(x):
    return np.sum(np.abs(x + 0.5)**2)

def F7(x):
    x = np.array(x)
    dim = len(x)
    return np.sum(np.arange(1, dim+1) * (x**4)) + np.random.rand()

def F8(x):
    x = np.array(x)
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    x = np.array(x)
    dim = len(x)
    return np.sum(x**2 - 10*np.cos(2*np.pi*x)) + 10*dim

def F10(x):
    x = np.array(x)
    dim = len(x)
    return (-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/dim))
            - np.exp(np.sum(np.cos(2*np.pi*x))/dim)
            + 20 + np.e)

def F11(x):
    x = np.array(x)
    dim = len(x)
    return (np.sum(x**2)/4000
            - np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1))))
            + 1)

def F12(x):
    x = np.array(x)
    dim = len(x)
    term1 = 10 * (np.sin(np.pi * (1 + (x[0]+1)/4)))**2
    term2 = np.sum(((x[:-1]+1)/4)**2 *
                   (1 + 10*(np.sin(np.pi*(1+(x[1:]+1)/4)))**2))
    term3 = ((x[-1]+1)/4)**2
    return (np.pi/dim)*(term1 + term2 + term3) + np.sum(Ufun(x,10,100,4))

def F13(x):
    x = np.array(x)
    dim = len(x)
    term1 = (np.sin(3*np.pi*x[0]))**2
    term2 = np.sum((x[:-1]-1)**2 *
                   (1 + (np.sin(3*np.pi*x[1:]))**2))
    term3 = (x[-1]-1)**2 * (1 + (np.sin(2*np.pi*x[-1]))**2)
    return 0.1*(term1 + term2 + term3) + np.sum(Ufun(x,5,100,4))

def F14(x):
    x = np.array(x)
    aS = np.array([
        [-32,-16,0,16,32]*5,
        [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5
    ])
    bS = np.array([np.sum((x - aS[:,j])**6) for j in range(25)])
    return (1/500 + np.sum(1/(np.arange(1,26) + bS)))**(-1)

def F15(x):
    x = np.array(x)
    aK = np.array([.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246])
    bK = 1/np.array([.25,.5,1,2,4,6,8,10,12,14,16])
    return np.sum((aK - ((x[0]*(bK**2 + x[1]*bK)) /
                         (bK**2 + x[2]*bK + x[3])))**2)

def F16(x):
    x1, x2 = x[0], x[1]
    return (4*x1**2 - 2.1*x1**4 + (x1**6)/3
            + x1*x2 - 4*x2**2 + 4*x2**4)

def F17(x):
    x1, x2 = x[0], x[1]
    return ((x2 - (5.1/(4*np.pi**2))*x1**2 + (5/np.pi)*x1 - 6)**2
            + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10)

def F18(x):
    x1, x2 = x[0], x[1]
    return ((1 + (x1+x2+1)**2 *
            (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
            *
            (30 + (2*x1 - 3*x2)**2 *
            (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)))

def F19(x):
    x = np.array(x)
    aH = np.array([[3,10,30],[.1,10,35],[3,10,30],[.1,10,35]])
    cH = np.array([1,1.2,3,3.2])
    pH = np.array([
        [.3689,.117,.2673],
        [.4699,.4387,.747],
        [.1091,.8732,.5547],
        [.03815,.5743,.8828]
    ])
    o = 0
    for i in range(4):
        o -= cH[i]*np.exp(-np.sum(aH[i]*(x - pH[i])**2))
    return o

def F20(x):
    x = np.array(x)
    aH = np.array([
        [10,3,17,3.5,1.7,8],
        [.05,10,17,.1,8,14],
        [3,3.5,1.7,10,17,8],
        [17,8,.05,10,.1,14]
    ])
    cH = np.array([1,1.2,3,3.2])
    pH = np.array([
        [.1312,.1696,.5569,.0124,.8283,.5886],
        [.2329,.4135,.8307,.3736,.1004,.9991],
        [.2348,.1415,.3522,.2883,.3047,.6650],
        [.4047,.8828,.8732,.5743,.1091,.0381]
    ])
    o = 0
    for i in range(4):
        o -= cH[i]*np.exp(-np.sum(aH[i]*(x - pH[i])**2))
    return o

def _F21_core(x, m):
    x = np.array(x)
    aSH = np.array([
        [4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],
        [3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],
        [6,2,6,2],[7,3.6,7,3.6]
    ])
    cSH = np.array([.1,.2,.2,.4,.4,.6,.3,.7,.5,.5])
    o = 0
    for i in range(m):
        o -= 1 / (np.sum((x - aSH[i])**2) + cSH[i])
    return o

def F21(x):
    return _F21_core(x, 5)

def F22(x):
    return _F21_core(x, 7)

def F23(x):
    return _F21_core(x, 10)