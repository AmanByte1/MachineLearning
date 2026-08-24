
# print(text.lower())
t="AMA"
t2="AMAMAq"

l=0
r=len(t2)-1

while l<r:
    if t2[l]!=t2[r]:
        print("Not Palindrome")
        break   
    
    l+=1
    r-=1
else:
     print("Palindrome")
    

    
