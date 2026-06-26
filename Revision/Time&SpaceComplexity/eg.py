num=[10,20,30,40]
print(num[2])
# Same speed even a 1000 num o1

n=[1,2,3,4,5]
for num in n:
    print(num)
# more elements more work or iterations  O(n)

n1=[1,2,3,4]

for i in n1:
    for j in n1:
        print(i,j)

# O(n^2) - n = 3 → 9 operations | n = 100 → 10,000 operations



# O(log n) — Logarithmic Time

# This one is very important but less intuitive.

# Imagine you're searching a word in a dictionary.

# You don't start from page 1.

# You open the middle.

# If your word comes before it, you ignore half the dictionary.

# If it comes after, you ignore the other half.

# Each step cuts the remaining work roughly in half.

# This is O(log n).

# You'll see this with binary search.