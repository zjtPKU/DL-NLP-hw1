import time
import random

# 造一些元素
def generate_nested_list(num_sublists, sublist_size):
    return [[random.randint(0, 100) for _ in range(sublist_size)] for _ in range(num_sublists)]


def flatten_with_for_loop(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def flatten_with_list_comprehension(nested_list):
    return [item for sublist in nested_list for item in sublist]


input_scales = [10**3, 10**4, 10**5, 10**6, 10**7,10**8]
sublist_size = 1000
for_loop_times = []
list_comprehension_times = []


for scale in input_scales:
    nested_list = generate_nested_list(scale // sublist_size, sublist_size)

    print(f"\nTesting for input size: {scale}")
    start = time.time()
    flatten_with_for_loop(nested_list)
    end = time.time()
    print(f"For-loop method: {end - start} seconds")
    for_loop_times.append(end - start)
    
    start = time.time()
    flatten_with_list_comprehension(nested_list)
    end = time.time()
    list_comprehension_times.append(end - start)
    
    print(f"List comprehension method: {end - start} seconds")

    # plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(input_scales, for_loop_times, label='For-loop method', marker='o')
plt.plot(input_scales, list_comprehension_times, label='List comprehension method', marker='s')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Input size (number of elements)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Performance Comparison: For-loop vs List Comprehension', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('performance_comparison.png')
# 显示图表
plt.show()
