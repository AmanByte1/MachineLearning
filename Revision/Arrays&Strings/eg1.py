n=[1,2,3,4,5]

l=0
r=len(n)-1

while l<r:
    n[l],n[r]=n[r],n[l]
    l+=1
    r-=1
print(n)

