l=[1,22,34,5,65,32,234,5]
ln=len(l)
r=0
n=[]
for num in l:
    if num > r:
        r2=r
        r = num
    elif num > r2 and num != r:
        
        r2 = num
print(r2)
print(r)
    
        