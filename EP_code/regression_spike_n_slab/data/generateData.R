
set.seed(1)

beta <- 10
p0 <- 0.5
v1 <- 1
rho <- 0.5

# We generate the data

nSimulations <- 10
for (i in 1 : nSimulations) {

	nFeatures <- 10
	nSamples <- 5

	# We sample the coefficients

	z <- runif(nFeatures) < p0

	w <- rnorm(nFeatures) * z

	# We generate the training data

	Xtrain <- matrix(rnorm(nFeatures * nSamples), nSamples, nFeatures)
	Ytrain <- as.double(Xtrain %*% w) + sqrt(1 / beta) * rnorm(nSamples)

	# We generate the test data

	nSamples <- 1000
	Xtest <- matrix(rnorm(nFeatures * nSamples), nSamples, nFeatures)
	Ytest <- as.double(Xtest %*% w) + sqrt(1 / beta) * rnorm(nSamples)

	write.table(Xtrain, paste("Xtrain", i, ".txt", sep = ""), col.names = F, row.names = F)
	write.table(Ytrain, paste("Ytrain", i, ".txt", sep = ""), col.names = F, row.names = F)
	write.table(Xtest, paste("Xtest", i, ".txt", sep = ""), col.names = F, row.names = F)
	write.table(Ytest, paste("Ytest", i, ".txt", sep = ""), col.names = F, row.names = F)
	write.table(t(w), paste("w", i, ".txt", sep = ""), col.names = F, row.names = F)
	write.table(t(z), paste("z", i, ".txt", sep = ""), col.names = F, row.names = F)

	print(i)
}

write.table(rho, "rho.txt", col.names = F, row.names = F)
write.table(p0, "p0.txt", col.names = F, row.names = F)
write.table(v1, "v1.txt", col.names = F, row.names = F)
write.table(beta, "beta.txt", col.names = F, row.names = F)
write.table(nSimulations, "nSimulations.txt", col.names = F, row.names = F)
