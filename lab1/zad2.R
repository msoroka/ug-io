# ZADANIE 2
# a)
getwd()
setwd("/Users/msoroka/Desktop/uczelnia/sem 7/io/lab1")
getwd()

# b)
b <- read.csv(file="osoby.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)
View(b)

# c)
View(b$imie)

# d)
d <- subset(b, grepl("k", b$plec))
View(d)

# e)
e <- subset(b, 50 < b$wiek & grepl("m", b$plec))
write.csv(e, file = "osoby2.csv")
View(e)

# f)
f <- mean(b$wiek)

# g)
b$wypłata <- round(runif(nrow(b), min=2000, max=5000))
View(b)

# h)
b[nrow(b) + 1,] = list('Jan', 'Kowalski',"m",30,3000)
View(b)
write.csv(b, file = "osoby3.csv")
