import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.rc('font', family='serif')

f2 = lambda f1 : -((f1 /100)**0.5 - 1)**2
F1_a,F1_b = np.linspace(1,16,300),np.linspace(36,81,300)
F2_a,F2_b = f2(F1_a),f2(F1_b)
plt.plot(F1_a, F2_a, linewidth=2.0, color='blue', label="Pareto-front")
plt.plot(F1_b, F2_b, linewidth=2.0, color='blue')

plt.xlabel("$f_1$", fontsize=14)
plt.ylabel("$f_2$", fontsize=14)


plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),fancybox=True, shadow=False)
plt.tight_layout()
plt.savefig("pateto.png", dpi=300, bbox_inches='tight')

