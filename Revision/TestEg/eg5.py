def rd(n):
    u=[]
    for i in n:
        if i not in u:
            u.append(i)
    return u
print(rd([1,2,3,3,4,4,5,5,6,6,6,7,77]))