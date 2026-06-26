n=[1,2,3,3,4,4,5,5,6,6,6,7,77]

f={}

for i in n:
    f[i]=f.get(i,0)+1

print(f)