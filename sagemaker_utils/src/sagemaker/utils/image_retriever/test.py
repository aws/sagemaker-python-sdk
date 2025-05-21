a = 1
b = 2
print(locals())
args = dict(locals())
print(args)
locals()["a"] = 3
print(a)
