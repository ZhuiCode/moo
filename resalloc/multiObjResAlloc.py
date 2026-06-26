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


#资源分配主要涉及到内容
#1.依据任务需求确定，所需要的排他性资源，所需的非排他性资源
#   1.1 依据任务需求，确定任务类型，任务类型具体可依据对数据的诉求划分为：数据独立型和非独立性
#   1.1.1 数据独立性任务：定义为该任务为最小任务可独立处理属于自己的数据，
#   1.1.2 数据非独立性任务：定义为该任务需要依赖其他任务的数据才能完成自己的任务，具有前后依赖
#    1.1.2.1  对非独立性任务需要进行任务划分和切割，做好任务的前后依赖关系，确保任务的顺利完成
#   1.2 形成任务集合，按照任务集合以及相互的依赖关系，确定资源的分配
#2.资源分配，确定卫星集合：
# 2.1 优先分配排他性资源，进一步缩小卫星集合 
# 2.2 依据卫星集合开展轨道周期预估，进一步锁定卫星
#   2.2.1 确定星内节点的资源是否匹配，如存储、算力资源；
# 2.3 通信质量估计：该内容是属于实际建立连接过程中才会涉及到，是否需要在前期规划时锁定，还需要进一步确认
# 2.4 
#3.非排他性资源的分配：
#轨道周期计算


#通信质量估计


#资源需求数学模型

#供电数学模型

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
approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

weights = np.array([0.2, 0.8])
#nF是指将目标函数值进行归一化处理后的结果，归一化的目的是为了消除不同目标函数之间的量纲
#差异，使得它们在优化过程中具有相同的权重，从而更好地反映出各个目标函数的重要性
nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
#伪权重的作用是将多目标优化问题转化为单目标优化问题，通过加权求和的方
# 式将多个目标函数合并成一个目标函数，从而简化优化过程
i = PseudoWeights(weights).do(nF)


plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)


plt.title("Design Space")
plt.savefig("design_space2.png", dpi=300)