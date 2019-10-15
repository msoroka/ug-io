# ZADANIE 4
# a)
losuj <- function(a,b) {
  return(round(runif(1, a, b)))
}

losuj(1,5)
losuj(10,50)

# b)
standaryzuj <- function(v) {
  return(scale(v))
}

standaryzuj(c(1,2,3,4))

# c)
normalizuj <- function(v) {
  return(nor.min.max(v))
}

normalizuj(c(1,2,3,4))

