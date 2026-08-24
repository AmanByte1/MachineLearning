def cc(t):
    f={}
    for c in t:
        if c in f:
            f[c]+=1
        else:
            f[c]=1
    return f
print(cc("hello world"))