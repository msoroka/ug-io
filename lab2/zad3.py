import random
import matplotlib.pyplot as plt
from collections import Counter

v = [random.randint(0, 20) for _ in range(100)]
counted = sorted(Counter(v).items())
plt.style.use('ggplot')

x = []
numbers = []

for i in counted:
    x.append(i[0])
    numbers.append(i[1])

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, numbers, color='green')
plt.xlabel("Cyfra")
plt.ylabel("Liczba wystąpień")
plt.title("Liczba wystąpień danej liczby w wektorze")
plt.xticks(x_pos, x)
plt.show()
