import numpy  as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl=np.array([-2, -2]), xu=np.array([2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 *(x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1= 2 *(x[0]-0.1)*(x[0]-0.9)/0.18
        g2 = - 20 *(x[0]-0.4)*(x[0]-0.6)/4.8
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

problem = MyProblem()


algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)
#get_termination函数设置优化的终止条件，这里设置为当算法运行40代时终止
termination = get_termination("n_gen", 40)
# minimize函数执行优化过程，返回优化结果res，其中problem表示优化问题，algorithm表示优化算法，termination表示终止条件，seed设置随机数种子，save_history表示是否保存优化历史，verbose表示是否输出优化过程中的
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
# X表示设计空间中的解，F表示目标函数值，res.X和res.F分别返回设计空间中的解和对应的目标函数值
X = res.X
F = res.F
#xl, xu 表示设计空间的下界和上界，problem.bounds()函数返回设计空间的边界值 
xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
#plt.scatter设置散点图，X[:, 0]表示x轴数据，X[:, 1]表示y轴数据，s设置点的大小，facecolors设置点的填充颜色，edgecolors设置点的边框颜色
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
# plt.xlim设置x轴范围，plt.ylim设置y轴范围
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.title("Design Space")
plt.savefig("design_space.png", dpi=300)