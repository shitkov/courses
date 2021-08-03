import random

import statistics

rand_list = [random.randint(10, 20) for _ in range(random.randint(10,15))]

rand_list.sort()

print(statistics.median(rand_list))

if len(rand_list) % 2 == 1:
    ans = rand_list[int(len(rand_list) / 2)]
else:
    ans = (rand_list[int(len(rand_list)/2)] + rand_list[int(len(rand_list)/2) - 1]) / 2

print(ans)