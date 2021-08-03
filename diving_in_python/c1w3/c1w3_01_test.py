from solution import FileReader

path = '/Users/kitaiskypanda/work/python/c1w3/test.txt'

reader = FileReader(path)
text = reader.read()
print(text)
