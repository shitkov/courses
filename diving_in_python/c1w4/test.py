from solution import File

path_to_file1 = '/Users/kitaiskypanda/work/python/c1w4/test11.txt'
path_to_file2 = '/Users/kitaiskypanda/work/python/c1w4/test22.txt'
test1 = File(path_to_file1)
test2 = File(path_to_file2)

new = test1 + test2

print(test1.read())
