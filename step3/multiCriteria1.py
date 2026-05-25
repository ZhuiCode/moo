import numpy  as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.mcdm.pseudo_weights import PseudoWeights


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl=np.array([-2, 2]), xu=np.array([2, 2]))

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

termination = get_termination("n_gen", 40)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
X = res.X
F = res.F
#F.min(axis=0)函数返回F矩阵每列的最小值，F.max(axis=0)函数返回F矩阵每列的最大值，approx_ideal和approx_nadir分别表示近似理想点和近似nadir点
#axis=0表示按列计算最小值和最大值，得到的结果是一个包含每列最小值和最大值的一维数组
fl = F.min(axis=0)
fu = F.max(axis=0)

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)
#np.array函数将列表转换为numpy数组，weights表示权重向量，
weights = np.array([0.2, 0.8])
#PseudoWeights(weights).do(nF)函数根据权重向量计算伪权重，nF表示归一化后的目标函数值矩阵
nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
#PseudoWeights(weights).do(nF)函数根据权重向量计算伪权重，nF表示归一化后的目标函数值矩阵
#伪权重的作用是将多目标优化问题转化为单目标优化问题，通过加权求和的方式将多个目标函数合并成一个目标函数，从而简化优化过程
i = PseudoWeights(weights).do(nF)



print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")
xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Design Space")
plt.savefig("design_space.png", dpi=300)