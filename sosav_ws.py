#coding:utf-8
from myutilities3_slim import *

from pylab import *
seterr(divide='raise', invalid='raise')
#seterr(divide='warn', invalid='warn')
#seterr(divide='ignore', invalid='ignore')

from collections import defaultdict
from pulp import *

import functools
print = functools.partial(print, flush=True)

import pickle

import sys
import time

class SOSAV_WS:
    def __init__(s, name="SOSAV-WS", printmode=1):
        s.py_start_time = time.time()
        s.PROB = LpProblem(name, LpMinimize)
        if printmode:
            s.print = print
        else:
            #printしない
            s.print = lambda *x: None
    
    def set_data(s, scenario="mh", RHO=1, ALPHA_T=5, ALPHA_D=1, ALPHA_N=10, ALPHA_C=10, PROBLEM_SCALE=1, POP=1000, seed=1, delay_max=10, DAY=-1, start_time=8, end_time=10, demand_aggregate_width=6, flat_demand=False):
        s.print("Defining data...")
        #所与変数定義
        s.print(" Scenario name:", scenario)
        
        s.flat_demand = flat_demand
        s.demand_aggregate_width = demand_aggregate_width
        
        if scenario == "mh":
            #NYタクシーデータ，マンハッタン
            
            s.ALPHA_T = ALPHA_T
            s.ALPHA_D = ALPHA_D
            s.ALPHA_N = ALPHA_N
            s.ALPHA_C = ALPHA_C
            
            network_data = readcsv("mh/edges_id_mh_slim.csv", "auto")
            demand_data = readcsv("mh/odtable_out.csv", "auto")
            
            tod_start = int(12*8)
            tod_end = int(12*9)
            tod_mid = int(12*8.5)
            tod_flush_end = int(12*10)
            number_of_timesteps = tod_flush_end-tod_start
            half_timestep = int(number_of_timesteps/4)
            
            s.NODES = []
            s.LINKS = []
            for e in network_data:
                if e[0] not in s.NODES:
                    s.NODES.append(e[0])
                if e[1] not in s.NODES:
                    s.NODES.append(e[1])
                if (e[0], e[1]) not in s.LINKS:
                    s.LINKS.append((e[0], e[1]))
                if (e[1], e[0]) not in s.LINKS:
                    s.LINKS.append((e[1], e[0]))
            
            node_veryhigh = [163,162,161,230,43,100,186,164,234,103,12,88,87,261,13,209]
            node_high = [143,142,237,229,48,246,68,170,233,90,45,231,113,211,144]
            node_mid = [244,24,151,238,239,236,263,262,140,202,50,107,137,79,148,125,158,249,114,232,4]
            node_low = [128,127,243,120,116,42,152,166,41,74,194,75,224]
            c_veryhigh = 20
            c_high = 10
            c_mid = 2
            c_low = 0.35
            s.c_node = {i:c_mid for i in s.NODES}
            s.c_link_coef = 4
            for i in s.NODES:
                if i in node_veryhigh:
                    s.c_node[i] = c_veryhigh
                elif i in node_high:
                    s.c_node[i] = c_high
                elif i in node_mid:
                    s.c_node[i] = c_mid
                elif i in node_low:
                    s.c_node[i] = c_low
            
            
            s.TIMES = [i for i in range(number_of_timesteps)]
            
            s.RHO = RHO
            
            MU_default = 20
            KAPPA_default = 20
            s.MU_default = MU_default
            s.KAPPA_default = KAPPA_default
            
            s.MU_min = MU_default/5
            s.KAPPA_min = KAPPA_default/5
            s.MU_max = MU_default*2
            s.KAPPA_max = KAPPA_default*2
            
            s.delay_max = int(number_of_timesteps/2)
            
            s.M = defaultdict(lambda : 0)
            #s.M[orig, dest, depart t]
            from numpy import random
            for l in demand_data:
                if l[0] in s.NODES and l[1] in s.NODES and tod_start <= l[2] < tod_end:
                    if l[2] < tod_mid:
                        s.M[l[0], l[1], 0] += l[3]
                    else:
                        s.M[l[0], l[1], half_timestep] += l[3]
        
        if scenario == "1d_city":
            #一次元都市の通勤
            
            s.ALPHA_T = ALPHA_T
            s.ALPHA_D = ALPHA_D
            s.ALPHA_N = ALPHA_N
            s.ALPHA_C = ALPHA_C
            
            PROBLEM_SCALE = PROBLEM_SCALE
            number_of_cities = 10*PROBLEM_SCALE
            number_of_timesteps = 20*PROBLEM_SCALE
            s.NODES = [i for i in range(1, number_of_cities+1)]
            
            s.LINKS = [(i, i+1) for i in range(1, number_of_cities)] + [(i+1, i) for i in range(1, number_of_cities)]
            
            s.c_node = {i:1 for i in s.NODES}
            s.c_link_coef = 4
            
            s.TIMES = [i for i in range(number_of_timesteps)]
            
            s.RHO = RHO
            
            MU_default = 20
            KAPPA_default = 50
            s.MU_default = MU_default
            s.KAPPA_default = KAPPA_default
            
            s.MU_min = MU_default
            s.KAPPA_min = KAPPA_default
            s.MU_max = MU_default*4
            s.KAPPA_max = KAPPA_default*4
            
            s.delay_max = number_of_timesteps
            
            s.POP = POP
            r_ave = 4*PROBLEM_SCALE
            r_std = 3*PROBLEM_SCALE
            s_ave = 7*PROBLEM_SCALE
            s_std = 3*PROBLEM_SCALE
            
            s.M = {}
            from numpy import random
            if seed != 0:
                random.seed(seed)
            while sum(list(s.M.values())) < POP:
                r = int(random.normal(r_ave, r_std))
                ss = int(random.normal(s_ave, s_std))
                if r != ss and r in s.NODES and ss in s.NODES:
                    if (r,ss,0) in s.M.keys():
                        s.M[r,ss,0] += 5
                    else:
                        s.M[r,ss,0] = 5
        
        s.print("  total demand types:", len(s.M.values()))
        s.print("  total demand:", sum([a for a in s.M.values()]))
        s.formulate()
    
    def formulate(s):
        s.print("Formulating problem...")
        
        ##########################################################################
        #変数定義
        s.print(" Defining parameters...")
        
        s.LINKS += [(i,i) for i in s.NODES]

        s.ORIG = [(0,i) for i in s.NODES]
        s.DEST = [(i,0) for i in s.NODES]

        s.NEXT = {}
        for i in s.NODES:
            s.NEXT[i] = []
            for i0,j in s.LINKS:
                if i0 == i:
                    s.NEXT[i].append(j)
        
        s.ODTIME = []
        for r,ss,k in s.M.keys():
            if (ss,k) not in s.ODTIME:
                s.ODTIME.append((ss,k))
        ODTIME_full = []
        for r,ss,k in s.M.keys():
            ODTIME_full.append((r,ss,k))
        
        s.x = {}
        for t in s.TIMES:
            for i,j in s.LINKS:
                if i != j:
                    s.x[i,j,t] = LpVariable("x_%d,%d_%d"%(i,j,t), 0, None)
                else:
                    s.x[i,j,t] = LpVariable("x_%d,%d_%d"%(i,j,t), 0, None)
        
        for i,j in s.ORIG:
            s.x[i,j,s.TIMES[0]] = LpVariable("x_%d,%d_%d"%(i,j,t), 0, None)
        for i,j in s.DEST:
            s.x[i,j,s.TIMES[-1]] = LpVariable("x_%d,%d_%d"%(i,j,t), 0, None)
        
        s.y = {}
        for t in s.TIMES:
            for i,j in s.LINKS+s.DEST:
                for ss,k in s.ODTIME:
                    s.y[ss,i,j,k,t] = LpVariable("y_%d,%d,%d_%d,%d"%(ss,i,j,k,t), 0, None)
        for r,ss,k in ODTIME_full:
            if s.flat_demand:
                for kk in range(s.demand_aggregate_width):
                    s.y[ss,0,r,k,k+kk] = LpVariable("y_%d,%d,%d_%d,%d"%(ss,0,r,k,k+kk), 0, None)
            else:
                s.y[ss,0,r,k,k] = LpVariable("y_%d,%d,%d_%d,%d"%(ss,0,r,k,k), 0, None)
        
        s.print("  total SAV links:", len(s.x.keys()))
        s.print("  total pax links:", len(s.y.keys()))
        
        s.mu = {}
        for i,j in s.LINKS:
            if i != j:
                s.mu[i,j] = LpVariable("mu_%d,%d"%(i,j), 0, None)

        s.kappa = {}
        for i in s.NODES:
            s.kappa[i] = LpVariable("kappa_%d"%(i), 0, None)
        
        s.T = LpVariable("T", 0, None)
        s.D = LpVariable("D", 0, None)
        s.N = LpVariable("N", 0, None)
        s.C = LpVariable("C", 0, None)
        
        ##########################################################################
        #制約条件定義
        s.print(" Defining constraints...")
        s.print("  Objectives...")
        
        s.PROB += lpSum([s.y[ss,i,j,k,t] for ss,k in s.ODTIME for i,j in s.LINKS for t in s.TIMES]) == s.T
        #total travel time
        
        s.PROB += lpSum([s.x[i,j,t] for i,j in s.LINKS for t in s.TIMES if i != j]) == s.D
        #total distance traveled by vehicles
        
        s.PROB += lpSum([s.x[0,i,s.TIMES[0]] for i in s.NODES]) == s.N
        #total number of vehicles
        
        s.PROB += lpSum([s.c_link_coef*s.c_node[i]*(s.mu[i,j]-s.MU_min) for i,j in s.LINKS if i != j]) + lpSum([s.c_node[i]*(s.kappa[i]-s.KAPPA_min) for i in s.NODES]) == s.C
        #total construction cost
        
        s.print("  Links...")
        for t in s.TIMES:
            for i in s.NODES:
                if t != s.TIMES[0] and t != s.TIMES[-1]:
                    s.PROB += lpSum([s.x[j,i,t-1] for j in s.NEXT[i]]) - lpSum([s.x[i,j,t] for j in s.NEXT[i]]) == 0
                    #vehicle conservation
                
                if t == s.TIMES[0]:
                    s.PROB += - lpSum([s.x[i,j,t] for j in s.NEXT[i]]) + s.x[0,i,t] == 0
                    #vehicle conservation initial
                
                if t == s.TIMES[-1]:
                    s.PROB += lpSum([s.x[j,i,t-1] for j in s.NEXT[i]]) - s.x[i,0,t] == 0
                    #vehicle conservation end
                
                for ss,k in s.ODTIME:
                    if t != s.TIMES[0] and t != s.TIMES[-1]:
                        try:
                            s.PROB += lpSum([s.y[ss,j,i,k,t-1] for j in s.NEXT[i]]) - lpSum([s.y[ss,i,j,k,t] for j in s.NEXT[i]]) + s.y[ss,0,i,k,t] - s.y[ss,i,0,k,t] == 0
                        except KeyError:
                            s.PROB += lpSum([s.y[ss,j,i,k,t-1] for j in s.NEXT[i]]) - lpSum([s.y[ss,i,j,k,t] for j in s.NEXT[i]]) - s.y[ss,i,0,k,t] == 0
                        #passenger conservation
                    
                    if t == s.TIMES[0]:
                        try:
                            s.PROB += - lpSum([s.y[ss,i,j,k,t] for j in s.NEXT[i] if (ss,i,j,k,t) in s.y.keys()]) + s.y[ss,0,i,k,t] - s.y[ss,i,0,k,t] == 0
                        except KeyError:
                            s.PROB += - lpSum([s.y[ss,i,j,k,t] for j in s.NEXT[i] if (ss,i,j,k,t) in s.y.keys()]) - s.y[ss,i,0,k,t] == 0
                        #passenger conservation initial
                    
                    if t == s.TIMES[-1]:
                        try:
                            s.PROB += lpSum([s.y[ss,j,i,k,t-1] for j in s.NEXT[i] if (ss,j,i,k,t-1) in s.y.keys()]) + s.y[ss,0,i,k,t] - s.y[ss,i,0,k,t] == 0
                        except KeyError:
                            s.PROB += lpSum([s.y[ss,j,i,k,t-1] for j in s.NEXT[i] if (ss,j,i,k,t-1) in s.y.keys()]) - s.y[ss,i,0,k,t] == 0
                        #passenger conservation end
                
                for j in s.NEXT[i]:
                    if i != j:
                        s.PROB += lpSum([s.y[ss,i,j,k,t] for ss,k in s.ODTIME]) <= s.RHO*s.x[i,j,t]
                        #vehicle capacity
                        
                        s.PROB += s.x[i,j,t] <= s.mu[i,j]
                        #road capacity
                    
                    if i == j:
                        s.PROB += s.x[i,j,t] <= s.kappa[i]
                        #parking capacity
        
        s.print("  ODs...")
        for r,ss,k in ODTIME_full:
            if (ss,0,r,k,k) in s.y.keys():
                if s.flat_demand:
                    for kk in range(s.demand_aggregate_width):
                        s.PROB += s.y[ss,0,r,k,k+kk] == s.M[r,ss,k]/s.demand_aggregate_width
                else:
                    s.PROB += s.y[ss,0,r,k,k] == s.M[r,ss,k]
            #passenger origin
            
        for ss,k in s.ODTIME:
            s.PROB += lpSum([s.y[ss,ss,0,k,t] for t in s.TIMES if t <= k+s.delay_max]) == sum([s.M[rr,sss,kk] for rr,sss,kk in ODTIME_full if ss == sss and k == kk])
            #passenger destination
            
            for i in s.NODES:
                for t in s.TIMES:
                    if i != ss:
                        s.PROB += s.y[ss,i,0,k,t] == 0
                        #passenger destination
        
        for i,j in s.LINKS:
            t = s.TIMES[-1]
            if i != j:
                s.PROB += s.x[i,j,t] == 0
                #end of day trips
            
            for ss,k in s.ODTIME:
                s.PROB += s.y[ss,i,j,k,t] == 0
                #end of day trips
            
            if i != j:
                s.PROB += s.mu[i,j] <= s.MU_max
                s.PROB += s.mu[i,j] >= s.MU_min
                #link capacity
        
        for i in s.NODES:
            s.PROB += s.kappa[i] <= s.KAPPA_max
            s.PROB += s.kappa[i] >= s.KAPPA_min
            #node capacity
        
        ##########################################################################
        #目的関数定義
        s.print(" Defining objectives...")
        s.PROB += s.ALPHA_T*s.T + s.ALPHA_D*s.D + s.ALPHA_N*s.N + s.ALPHA_C*s.C
    
    def save_data(s, name):
        f = open(name, "wb")
        pickle.dump(s.PROB, f, protocol=-1)
        f.close()
    
    def read_data(s, name):
        s.print("\nRead pickled problem:", name)
        f = open(name, "rb")
        s.PROB = pickle.load(f)
        f.close()
    
    def change_objective_function(s, ALPHA_T=-1, ALPHA_D=-1, ALPHA_N=-1, ALPHA_C=-1):
        s.print("\nRe-defining objective function...")
        if ALPHA_T != -1:
            s.ALPHA_T = ALPHA_T
        if ALPHA_D != -1:
            s.ALPHA_D = ALPHA_D
        if ALPHA_N != -1:
            s.ALPHA_N = ALPHA_N
        if ALPHA_C != -1:
            s.ALPHA_C = ALPHA_C
        s.PROB += s.ALPHA_T*s.PROB.variablesDict()["T"] + s.ALPHA_D*s.PROB.variablesDict()["D"] + s.ALPHA_N*s.PROB.variablesDict()["N"] + s.ALPHA_C*s.PROB.variablesDict()["C"]
    
    def constrain_objective(s, T=-1, D=-1, N=-1, C=-1):
        s.print("\nConstraining objective function:")
        if T != -1:
            s.PROB += s.PROB.variablesDict()["T"] == T
            s.print("T = %.1f"%T)
        if D != -1:
            s.PROB += s.PROB.variablesDict()["D"] == D
            s.print("D = %.1f"%D)
        if N != -1:
            s.PROB += s.PROB.variablesDict()["N"] == N
            s.print("N = %.1f"%N)
        if C != -1:
            s.PROB += s.PROB.variablesDict()["C"] == C
            s.print("C = %.1f"%C)
    
    def solve(s, solver_name="cbc", timelimit=3600):
        s.print("Solving the problem...")

        s.print(" Number of decision variables:", s.PROB.numVariables())
        s.print(" Number of constraints:", s.PROB.numConstraints(), "\n")
        
        #s.PROB.writeLP("savproblem.lp")
        #s.var = s.PROB.variablesDict()
        
        if solver_name == "gurobi":
            solver = pulp.GUROBI_CMD()
        elif solver_name == "cbc":
            solver = pulp.PULP_CBC_CMD(timeLimit=timelimit)
        
        s.py_solve_start_time = time.time()
        
        s.PROB.solve(solver)
        
        s.py_end_time = time.time()
        
        s.print("Solver:", s.PROB.solver.path)
        s.print("Status:", LpStatus[s.PROB.status])
        s.print("Solution time:", s.PROB.solutionTime, ",", s.py_end_time-s.py_solve_start_time)
        s.print("Total computation time:", (s.py_end_time-s.py_start_time), "\n")
        
        if s.PROB.status != 1:
            return -1
        
        if "T" in s.__dict__.keys():        
            s.print("T: %.1f"%s.T.varValue, "\t(alpha=%.1f)"%s.ALPHA_T)
            s.print("D: %.1f"%s.D.varValue, "\t(alpha=%.1f)"%s.ALPHA_D)
            s.print("N: %.1f"%s.N.varValue, "\t(alpha=%.1f)"%s.ALPHA_N)
            s.print("C: %.1f"%s.C.varValue, "\t(alpha=%.1f)"%s.ALPHA_C)
        else:
            s.T = s.PROB.variablesDict()["T"]
            s.D = s.PROB.variablesDict()["D"]
            s.N = s.PROB.variablesDict()["N"]
            s.C = s.PROB.variablesDict()["C"]
            s.print("T: %.1f"%s.T.varValue)
            s.print("D: %.1f"%s.D.varValue)
            s.print("N: %.1f"%s.N.varValue)
            s.print("C: %.1f"%s.C.varValue)
    
    def analyze(s, name="", mapcsv="mh/map.csv"):
        q = {(i,j):0 for i,j in s.LINKS}
        p = {i: 0 for i in s.NODES}
        ppeak = {i: 0 for i in s.NODES}
        for i,j in s.LINKS:
            for t in s.TIMES:
                if i != j:
                    q[i,j] += s.x[i,j,t].varValue
                if i == j:
                    p[i] += s.x[i,i,t].varValue
                    if s.x[i,i,t].varValue > ppeak[i]:
                        ppeak[i] = s.x[i,i,t].varValue
        q_max = max(q.values())
        p_max = max(p.values())

        q_pax = {(i,j):0 for i,j in s.LINKS}
        p_pax = {i: 0 for i in s.NODES}
        demand_pax = {i: 0 for i in s.NODES}
        for (ss,i,j,k,t) in s.y.keys():
            if s.y[ss,i,j,k,t].varValue > 0 and i != 0 and j != 0:
                if i != j:
                    q_pax[i,j] += s.y[ss,i,j,k,t].varValue
                if i == j:
                    p_pax[i] += s.y[ss,i,j,k,t].varValue
        for (ss, r, t) in s.M.keys():
            demand_pax[ss] += s.M[ss,r,t]

        mus = {(i,j):0 for i,j in s.LINKS}
        kappas = {i: 0 for i in s.NODES}
        for i,j in s.LINKS:
            if i != j:
                mus[i,j] = s.mu[i,j].varValue
            kappas[i] = s.kappa[i].varValue
        
        MM = {}
        for l in readcsv(mapcsv, "auto"):
            MM[l[0]] = {
                "x": l[1],
                "y": l[2],
                "name": l[3],
                "q_pax": 0,
                "q": 0,
                "mu": 0
            }

        for i in p_pax.keys():
            MM[i]["p_pax"] = p_pax[i]
            MM[i]["demand_pax"] = demand_pax[i]
        for i,j in q_pax.keys():
            MM[i]["q_pax"] += q_pax[i,j]
        for i in p.keys():
            MM[i]["p"] = p[i]
        for i,j in q.keys():
            MM[i]["q"] += q[i,j]
        for i,j in s.LINKS:
            MM[i]["mu"] += mus[i,j]
            MM[i]["kappa"] = kappas[i]
        
        if name == "":
            name = f"{s.ALPHA_T}_{s.ALPHA_D}_{s.ALPHA_N}_{s.ALPHA_C}"
        f = open(f"dat/resMM_{name}.bin", "wb")
        pickle.dump(MM, f)
        f.close()
        
        tdt_pax = [0 for t in s.TIMES]
        tts_pax = [0 for t in s.TIMES]
        tdt = [0 for t in s.TIMES]
        tts = [0 for t in s.TIMES]

        for i,j in s.LINKS:
            for t in s.TIMES:
                for ss,k in s.ODTIME:
                    if i != j:
                        tdt_pax[t] += s.y[ss,i,j,k,t].varValue
                        tts_pax[t] += s.y[ss,i,j,k,t].varValue
                    if i == j:
                        tts_pax[t] += s.y[ss,i,j,k,t].varValue
        
        for i,j in s.LINKS:
            end_flag = 1
            for t in reversed(s.TIMES):
                if i != j:
                    tdt[t] += s.x[i,j,t].varValue
                    tts[t] += s.x[i,j,t].varValue
                if end_flag == 1:
                    if t < max(s.TIMES)-1 and s.x[i,j,t].varValue > s.x[i,j,t+1].varValue:
                        end_flag = 0
                if i == j and end_flag == 0:
                    tts[t] += s.x[i,j,t].varValue
        
        f = open(f"dat/resedie_{name}.bin", "wb")
        pickle.dump([tdt, tts, tdt_pax, tts_pax], f)
        f.close()
    
    def flow_analysis(s, mapcsv, visualize=0):
        node_loc = {}
        for l in readcsv(mapcsv, "auto"):
            node_loc[l[0]] = {"x": l[1], "y":l[2]}
        
        vis_links_x = defaultdict(lambda : 0)
        vis_links_y = defaultdict(lambda : 0)
        for i,j in s.LINKS:
            for t in s.TIMES:
                if s.x[i,j,t].varValue > 0:
                    vis_links_x[i,j,t] += s.x[i,j,t].varValue
        for (ss,i,j,k,t) in s.y.keys():
            if i != 0 and j != 0 and s.y[ss,i,j,k,t].varValue > 0:
                vis_links_y[i,j,t] += s.y[ss,i,j,k,t].varValue
        
        if visualuze:
            figure(figsize=(6, 12))
            for key in vis_links_x:
                if vis_links_x[key] > 0.2:
                    x_cord = [node_loc[key[0]]["x"], node_loc[key[1]]["x"]]
                    t_cord = [key[2], key[2]+1]
                    width = vis_links_x[key]
                    plot(x_cord, t_cord, "k-", linewidth=width)
        
            show()
        
            figure(figsize=(6, 12))
            for key in vis_links_y:
                if vis_links_y[key] > 0.2:
                    x_cord = [node_loc[key[0]]["x"], node_loc[key[1]]["x"]]
                    t_cord = [key[2], key[2]+1]
                    width = vis_links_y[key]
                    if vis_links_y[key] >=  vis_links_x[key]*s.RHO*0.9:
                        c = "k"
                    else:
                        c = "k"
                    plot(x_cord, t_cord, "-", c=c, linewidth=width)
            
            show()
    
    def time_series_analysis(s, savename, visualize=0):
        x_travel = [0 for t in s.TIMES]
        x_wait = [0 for t in s.TIMES]
        x_occupied = [0 for t in s.TIMES]
        y_travel = [0 for t in s.TIMES]
        y_wait = [0 for t in s.TIMES]
        for i,j,t in s.x.keys():
            if i != 0 and j != 0:
                if i != j:
                    x_travel[t] += s.x[i,j,t].varValue
                else:
                    x_wait[t] += s.x[i,j,t].varValue
        for ss,i,j,k,t in s.y.keys():
            if i != 0 and j != 0:
                if i != j:
                    x_occupied[t] += s.y[ss,i,j,k,t].varValue/s.RHO
                    y_travel[t] += s.y[ss,i,j,k,t].varValue
                else:
                    y_wait[t] += s.y[ss,i,j,k,t].varValue
        
        if visualize == 1:
            figure()
            subplot(211)
            plot(x_travel, "b-", label="traveling")
            plot(x_wait, "r-", label="waiting")
            plot(x_occupied, "g-", label="traveling with pax")
            xlabel("time")
            ylabel("# of SAVs")
            legend()
            grid()
            
            subplot(212)
            plot(y_travel, "b-", label="traveling")
            plot(y_wait, "r-", label="waiting")
            xlabel("time")
            ylabel("# of pax")
            legend()
            grid()
            
            savefig(savename+".png")
            close()
            
        writecsv(savename+".csv", [x_travel, x_wait, x_occupied, y_travel, y_wait])
    
    def visualize(s, top_view=0, infra=0, tsd=0):
        
        if top_view:
            q = {(i,j):0 for i,j in s.LINKS}
            p = {i: 0 for i in s.NODES}
            ppeak = {i: 0 for i in s.NODES}
            for i,j in s.LINKS:
                for t in s.TIMES:
                    if i != j and s.x[i,j,t].varValue > 0.1:
                        q[i,j] += s.x[i,j,t].varValue
                    if i == j and s.x[i,i,t].varValue > 0.1:
                        p[i] += s.x[i,i,t].varValue
                        if s.x[i,i,t].varValue > ppeak[i]:
                            ppeak[i] = s.x[i,i,t].varValue
            q_max = max(q.values())
            p_max = max(p.values())
            
            figure(figsize=(len(s.NODES),2), dpi=150)
            title(r"total traffic volume and parking $\Sigma_{t} x_{ij}^t$")
            for i,j in s.LINKS:
                if q[i,j] > 0:
                    if i > j:
                        plot([i,j], [0.1, 0.1], "b-", lw=q[i,j]/q_max*5)
                        text((i+j)/2, 0.2, int(q[i,j]), color="b", horizontalalignment="center", verticalalignment="bottom")
                    else:
                        plot([j,i], [-0.1, -0.1], "g-", lw=q[i,j]/q_max*5)
                        text((i+j)/2, -0.2, int(q[i,j]), color="g", horizontalalignment="center", verticalalignment="top")
                if i == j and p[i] > 0:
                    plot(i, 0, "ro", mew=0, ms=p[i]/p_max*10)
                    text(i, -0.2, int(p[i]), color="r", horizontalalignment="center", verticalalignment="top")
                    text(i, 0.2, "(%d)"%ppeak[i], color="r", horizontalalignment="center", verticalalignment="bottom")
                    
                        
            xlabel("location $i$")
            xlim([0, max(s.NODES)+1])
            xticks([i for i in range(0,len(s.NODES)+2)])
            ylim([-1,1])
            grid()
            
            
            tight_layout()
            #savefig("img/volume_parking.png")
            show()
            #close()
        
        if infra:
            figure(figsize=(len(s.NODES),2), dpi=150)
            title(r"infrastructure $\mu_{ij}, \kappa_i$")
            for i,j in s.LINKS:
                if i != j and s.mu[i,j].varValue > 0:
                    if i > j:
                        plot([i,j], [0.1, 0.1], "b-", lw=s.mu[i,j].varValue/s.MU_default*2)
                        text((i+j)/2, 0.2, int(s.mu[i,j].varValue), color="k", horizontalalignment="center", verticalalignment="bottom")
                    else:
                        plot([j,i], [-0.1, -0.1], "g-", lw=s.mu[i,j].varValue/s.MU_default*2)
                        text((i+j)/2, -0.2, int(s.mu[i,j].varValue), color="k", horizontalalignment="center", verticalalignment="top")
                if i == j and s.kappa[i].varValue > 0:
                    plot(i, 0, "ro", mew=0, ms=s.kappa[i].varValue/s.KAPPA_default*10)
                    text(i, -0.2, int(s.kappa[i].varValue), color="k", horizontalalignment="center", verticalalignment="top")
                    
            xlabel("location $i$")
            xlim([0, max(s.NODES)+1])
            xticks([i for i in range(0,len(s.NODES)+2)])
            ylim([-1,1])
            grid()
            
            tight_layout()
            #savefig("img/infra.png")
            show()
            #close()
            
        if tsd:
            figure(figsize=(10,len(s.TIMES)/3), dpi=150)
            subplot(122)
            title("vehicle flow $x_{ij}^t$")
            for i,j in s.LINKS:
                for t in s.TIMES:
                    if s.x[i,j,t].varValue > 0.1:
                        if i != j:
                            plot([i,j], [t,t+1], "k-", lw=s.x[i,j,t].varValue/s.MU_default*5)
                            if s.x[i,j,t].varValue >= s.mu[i,j].varValue*0.95:
                                plot([i,j], [t,t+1], "r-", lw=s.x[i,j,t].varValue/s.MU_default*5)
                        else:
                            plot([i,j], [t,t+1], "k--", lw=s.x[i,j,t].varValue/s.MU_default*5, dashes=(5, 1), zorder=-10)
                            if s.x[i,j,t].varValue >= s.kappa[i].varValue*0.95:
                                plot([i,j], [t,t+1], "r--", lw=s.x[i,j,t].varValue/s.MU_default*5, dashes=(5, 1), zorder=-10)
            for i in s.NODES:
                t = s.TIMES[0]
                if s.x[0,i,t].varValue > 0.1:
                    text(i, t, int(s.x[0,i,t].varValue), color="b", horizontalalignment="center", verticalalignment="top")
                t = s.TIMES[-1]
                if s.x[i,0,t].varValue > 0.1:
                    text(i, t, int(s.x[i,0,t].varValue), color="g", horizontalalignment="center", verticalalignment="bottom")
            xlabel("location $i$")
            xlim([0, max(s.NODES)+1])
            xticks([i for i in range(0,len(s.NODES)+2)])
            ylabel("time step $t$")
            ylim([s.TIMES[0]-1, s.TIMES[-1]+1])
            yticks([i for i in range(s.TIMES[0]-1,s.TIMES[-1]+2)])
            grid()
            
            subplot(121)
            title(r"aggregated traveler flow $\Sigma_{ss,k}$ $y_{ss,ij}^{k,t}$")
            yy = {}
            for i,j in s.LINKS+s.ORIG+s.DEST:
                for t in s.TIMES:
                    yy[i,j,t] = 0
                    for ss,k in s.ODTIME:
                        try:
                            yy[i,j,t] += s.y[ss,i,j,k,t].varValue
                        except KeyError:
                            pass
            for i,j in s.LINKS:
                for t in s.TIMES:
                    if yy[i,j,t] > 0.1:
                        if i != j:
                            plot([i,j], [t,t+1], "k-", lw=yy[i,j,t]/s.MU_default*5)
                            if yy[i,j,t] >= s.RHO*s.mu[i,j].varValue*0.95:
                                plot([i,j], [t,t+1], "r-", lw=yy[i,j,t]/s.MU_default*5)
                        else:
                            plot([i,j], [t,t+1], "k--", lw=yy[i,j,t]/s.MU_default*5, dashes=(5, 1), zorder=-10)
                        #text((3*i+j)/4, (3*t+t+1)/4, int(yy[i,j,t]), color="k", horizontalalignment="center", verticalalignment="top")
            for i in s.NODES:
                for t in s.TIMES:
                    if yy[0,i,t] > 0.1:
                        text(i, t, int(yy[0,i,t]), color="b", horizontalalignment="center", verticalalignment="top")
                    if yy[i,0,t] > 0.1:
                        text(i, t, int(yy[i,0,t]), color="g", horizontalalignment="center", verticalalignment="bottom")
            xlabel("location $i$")
            xlim([0, max(s.NODES)+1])
            xticks([i for i in range(0,len(s.NODES)+2)])
            ylabel("time step $t$")
            ylim([s.TIMES[0]-1, s.TIMES[-1]+1])
            yticks([i for i in range(s.TIMES[0]-1,s.TIMES[-1]+2)])
            grid()
            
            tight_layout()
            #savefig("img/all.png")
            show()
            #close()



if __name__ == "__main__":
    
    prob = SOSAV_WS()
    prob.set_data()
    prob.save_data("sosav_ws_mh.bin")
    #prob.read_data("sosav_ws_mh.bin")
    prob.solve("gurobi")
    #prob.visualize(top_view=1, tsd=1)