
i = 0
sparse_indices = 3
sparse_values = 1
default_value = 0
dense = []
dense[i] = (sparse_values if i == sparse_indices else default_value)

print (dense[0])