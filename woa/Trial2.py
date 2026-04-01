#WOA Optimization of Seagrass Restoration
print("Import relevant packages")
import numpy as np
import v_seagrass2 #Used a different more efficient version of the seagrass function for this trial
import v_coral
import v_depth
import random
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
import argparse
from src.whale_optimization import WhaleOptimization
#####################################
#Function to build environment data
def build_env():
    #Define 2 vectors of lat and lon values
    latq = np.linspace(24, 32, 600)
    lonq = np.linspace(-86, -80, 600)
    #Make our meshgrid
    LonGrid, LatGrid = np.meshgrid(lonq,latq)
    print('Grid Made...Starting to collect seagrass data...')
    #Call seagrass function... get coverage values for each point in the grid
    seagrass_coverage = v_seagrass2.seagrass(LonGrid, LatGrid)
    print('Seagrass Collected!  Now for Depth...')
    #Call depth function... get depth values for each point in the grid
    depth = v_depth.depth(LonGrid, LatGrid)
    print('Depth Collected!')
    #Fun metric useful for debugging
    print(v_depth.depth)
    print('Setup DONE!')
    return(seagrass_coverage, depth, LonGrid, LatGrid)

#Here is our optimization function
def planting(X,Y,LonGrid, LatGrid, seagrass_coverage, depth):
    #Find the nearest grid point to the given X,Y coordinates
    y = np.argmin(np.abs(LatGrid[:,0] - Y))  # nearest row
    x = np.argmin(np.abs(LonGrid[0,:] - X))  # nearest column
    #Collect seagrass and depth values at that grid point
    sc = seagrass_coverage[y,x]
    d = depth[y,x]
    #Apply our planting criteria to determine viability score for that point
    good_depth = (d > 1) and (d < 3)
    if good_depth and (sc == np.nan):
        return 1
    if good_depth and (sc == "51 - 100%"):
        return 0.25
    if good_depth and (sc == "90 - 100%"):
        return 0.5
    if good_depth and (sc == "1 - 89%"):
        return 0.55
    if good_depth and (sc == "10 - 50%"):
        return 0.7
    if good_depth and (sc == "Continuous"):
        return 0
    if good_depth and (sc == "Discontinuous"):
        return 0.5
    if good_depth and (sc == "<50%"):
        return 0.75
    if good_depth and (sc == "Unknown"):
        return 0.5
    if good_depth and (sc == ""):
        return 0.5
    if good_depth and (sc == ">50%"):
        return 0.25
    if good_depth and (sc == "Continuous Seagrass"):
        return 0
    if good_depth and (sc == "Patchy (Discontinuous) Seagrass"):
        return 0.5
    if d > 3:
        return 0 #Too Deep
    if d <= 1:
        return 0 #Too Shallow
    if d == np.nan:
        return 0 #No Data on Depth
    else :
        return 0.5 #Default viability for areas not meeting other criteria
  
#Variables for optimization algorithm
def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nsols", type=int, default=100, dest='nsols', help='number of solutions per generation, default: 50')
    parser.add_argument("-ngens", type=int, default=30, dest='ngens', help='number of generations, default: 20')
    parser.add_argument("-a", type=float, default=2.0, dest='a', help='woa algorithm specific parameter, controls search spread default: 2.0')
    parser.add_argument("-b", type=float, default=0.5, dest='b', help='woa algorithm specific parameter, controls spiral, default: 0.5')
    parser.add_argument("-c", type=float, default=None, dest='c', help='absolute solution constraint value, default: None, will use default constraints')
    parser.add_argument("-func", type=str, default='planting', dest='func', help='function to be optimized, default: planting')
    parser.add_argument("-r", type=float, default=0.25, dest='r', help='resolution of function meshgrid, default: 0.25')
    parser.add_argument("-t", type=float, default=0.1, dest='t', help='animate sleep time, lower values increase animation speed, default: 0.1')
    parser.add_argument("-max", default=False, dest='max', action='store_true', help='enable for maximization, default: False (minimization)')

    args = parser.parse_args()
    return args

#####################################
#Main function to run optimization
def main():
    args = parse_cl_args() #Read command line options
    nsols = args.nsols #number of solutions per generation
    ngens = args.ngens #number of generations

    funcs = {'planting':planting} #map names to functions
    func_constraints = {'planting':1.0} #map constraints to functions (was in original code but not used now!)

    #Checks that you called function correctly!
    if args.func in funcs: #Checks did user define function right?
        func = funcs[args.func]
    else:
        print('Missing supplied function '+args.func+' definition. Ensure function defintion exists or use command line options.')
        return

    if args.c is None:
        if args.func in func_constraints:
            args.c = func_constraints[args.func]
        else:
            print('Missing constraints for supplied function '+args.func+'. Define constraints before use or supply via command line.')
            return


    #Defines search space as [-C, C] in each dimension
    constraints = [[-86,-80],[24,32]]

    opt_func = func #Store objective function

    b = args.b #parameter controlling spiral
    a = args.a #step size that decreases with generation
    a_step = a/ngens

    maximize = args.max #boolean (true if maximizing, false if minimizin)

    #Build environment data
    print('Building environment data...')
    seagrass_coverage, depth, LonGrid, LatGrid = build_env()
    #Initialize optimization algorithm
    print('Initializing optimization algorithm...')
    opt_alg = WhaleOptimization(opt_func, constraints, nsols, b, a, a_step, LonGrid, LatGrid, seagrass_coverage, depth, maximize)
    
    #Run optimization
    print('Running optimization...')
    for gen in range(ngens):
        opt_alg.optimize()
    
    #Save all solutions for each generation
    all_solutions = []
    for gen_positions in opt_alg.agent_paths:
        generation_data = []
        for agent_pos in gen_positions:
            x, y = agent_pos
            fitness = opt_func(x,y,LonGrid, LatGrid, seagrass_coverage, depth)
            generation_data.append((fitness, (x,y)))
        all_solutions.append(generation_data)

    # Save to pickle file!
    with open("all_solutions_with_values.pkl", "wb") as f:
        pickle.dump(all_solutions, f)

    # Load
    with open("all_solutions_with_values.pkl", "rb") as f:
        all_solutions_loaded = pickle.load(f)

    #Lets look at what we have...
    print(type(all_solutions_loaded))
    print("Number of Generations:", len(all_solutions_loaded))
    print('Fitness of Agent 0 in Generation 0:',all_solutions_loaded[0][0][0])
    print("Position of Agent 0 in Generation 0:",all_solutions_loaded[0][0][1])
    print('Longitude of Agent 0 in Generation 0:',all_solutions_loaded[0][0][1][0])
    print('Latitude of Agent 0 in Generation 0:',all_solutions_loaded[0][0][1][1])


if __name__ == '__main__':
    main()
