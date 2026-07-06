def c(n):
    if n==0:
        return
    print(n)
    c(n-1)
c(5)


def c(n):
    if n==0:
        return
    
    c(n-1)
    print(n)

c(5)
