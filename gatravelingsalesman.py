import random, matplotlib.pyplot as plt
cities = [random.sample(range(100), 2) for x in range(20)]
plt.plot(cities, 'xb')
plt.show()