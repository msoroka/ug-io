# ZADANIE 1
# a)
a = 45*678

# b)
x <- c(7, 4, 2, 0, 9)
y <- c(2, 1, 5, 3, 3)

# c)
c = x+y

# d)
d = x*y

# e)
e = t(x) %*% y

# f)
m1 <- matrix(c(0, 2, 1, 1, 6, 4, 5, 0, 3), nrow = 3, ncol = 3, byrow = TRUE)
m2 <- matrix(c(9, 8, 7, 1, 2, 7, 4, 9, 2), nrow = 3, ncol = 3, byrow = TRUE)
f = m1 %*% m2

# g)
v1 = 1
g1 = seq(v1, 100, 1)
g2 = c(1:100)

# h)
hsum = sum(g1)
havg = mean(g1)
hsd = sd(g1)

# i)
i = round(runif(20, min=0, max=50))
imin = min(i)
imax = max(i)
