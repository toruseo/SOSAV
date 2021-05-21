#coding:utf-8
from myutilities3_slim import *

from pylab import *
seterr(divide='raise', invalid='raise')
#seterr(divide='warn', invalid='warn')
#seterr(divide='ignore', invalid='ignore')

from sosav_ws import SOSAV_WS

from concurrent import futures
__spec__ = ""#"ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"


def solve(arg):
    pname, solver, D, N, C, rho = arg
    
    p2 = SOSAV_WS(printmode=0)
    #p2.read_data(pname)
    p2.set_data(scenario="mh", RHO=rho, ALPHA_T=100, ALPHA_D=D, ALPHA_N=N, ALPHA_C=C)
    #p2.change_objective_function(ALPHA_T=100, ALPHA_D=D, ALPHA_N=N, ALPHA_C=C)
    p2.solve(solver)
    
    print("######## solved")
    if p2.PROB.status == 1:
        print(" ".join(["%7.1f"%a.varValue for a in (p2.T, p2.D, p2.N, p2.C)]))
        p2.analyze()
        print("######## analyzed")
        r = [100, D, N, C, p2.T.varValue, p2.D.varValue, p2.N.varValue, p2.C.varValue]
        writecsv(f"dat/res_{100}_{D}_{N}_{C}.csv", [r])
        print("######## saved")
        return 100, D, N, C, p2.T.varValue, p2.D.varValue, p2.N.varValue, p2.C.varValue
    else:
        print(" ".join(["%7.1f"%a.varValue for a in (-1, D, N, C)]))
        r = [100, D, N, C, -1, -1, -1, -1]
        writecsv(f"dat/res_{100}_{D}_{N}_{C}.csv", [r])
        return 100, D, N, C, -1, -1, -1, -1


if __name__ == "__main__":
    
    rho = 5
    
    Ds = [1,10,100,1000,10000]
    Ns = [1,10,100,1000,10000]
    Cs = [1,10,100,1000,10000]
    
    solver = "gurobi"
    #solver = "cbc"
    max_workers = 40
    pname = "sosav_parallel.bin"
    
    #prob = SOSAV_WS(printmode=0)
    #prob.set_data(scenario="mh", RHO=rho, POP=1000)
    #prob.save_data(pname)
    
    args = [(pname, solver, D, N, C, rho) for D in Ds for N in Ns for C in Cs]
    
    res = []
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(len(args)):
            future = executor.submit(fn=solve, arg=args[i])
            res.append(future)
    
    out = [["T", "D", "N", "C", "T", "D", "N", "C"]]
    for r in res:
        rr = r.result()
        if -1 not in rr:
            out.append([rr[0], rr[1], rr[2], rr[3], rr[4], rr[5], rr[6], rr[7]])
    writecsv("res_1d.csv", out)
    