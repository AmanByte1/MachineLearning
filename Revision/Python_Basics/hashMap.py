m={}
m["a"]=1
m["b"]=2
m["d"]=1
m["d"]=2

print(m)

print(m)
print(m.get("d",6)+2)
m["d"]=m.get("d",4)+2
print(m)
n=[1,2,2,2,2,3,4,5,5,6,7,9]

# f={"a":1,"b":2}
# f[0]=f.get("b")
# print(f)
f={}

for num in n:
    f[num]=f.get(num,0)+1

print(f)
