import torch
import torch.nn as nn

'''
We can write alphaT = 1-BetaT <- this is done to make implementation easier, but you can still write it as 1-BetaT
Markov property allows us to write q(xt|x0) as a cumulative product of q(xt|xt-1) -> x0
q(xt|xt-1) = Normal(xt;sqrt(1-BetaT)xt-1,BetaTI)
Parametrization Trick can write a distribution Normal(u,var^2) -> u + sigma* epsilon
--> q(xt|xt-1) = sqrt(1-BetaT)xt-1 + sqrt(BetaT)*epsilon
-> #replacing with at now ->
q(xt|xt-1) = sqrt(alphaT)xt-1 + sqrt(1-alphaT)*epsilon <-This is the final version, however, we can write this to get at for any t value immediately by
this property
q(xt|xt-3) = sqrt(alphaT * AlphaT-1 * alphaT-2)xt-3 + sqrt(1-alphaT*alphaT-1*alphaT-2)*epsilon <- we can write this recursively for xt by plugging in
xt-1 and then xt-2 -> and so on for T length.
Demonstration of Derivation
Given two independent Gaussians N(0,1)
Z1 = A*epsilon1 Z2 = B*epsilon2 ->
Z1 + Z2 = sqrt(A^2+B^2)*epsilon -> Since Gaussian, it stays N(0,1)

q(xt|xt-1) = sqrt(alphaT)xt-1 + sqrt(1-alphaT)*epsilon1
q(xt-1|xt-2) = sqrt(alphaT-1)xt-2 + sqrt(1-alphaT-1)*epsilon2

-> we can write
q(xt|xt-2) = sqrt(alphaT)[sqrt(alphaT-1)*xt-2 + sqrt(1-alphaT-1)*epsilon2] + sqrt(1-alphaT)*epsilon1
= sqrt(alphaT)*sqrt(alphaT-1)*xt-2 + sqrt(alphaT)*sqrt(1-alphaT-1)*epsilon2 + sqrt(1-alphaT)*epsilon1
= sqrt(alphaT*alphaT-1)*xt-2 + sqrt(alphaT-alphaT*alphaT-1 + 1 -alphaT) epsilon <- Using gaussian property Z1 + Z2
= sqrt(alphaT * alphaT-1)*xt-2 + sqrt(1-alphaT*alphaT-1)*epsilon
Given this we can write
q(xt|xt-3) = sqrt(alphaT * AlphaT-1 * alphaT-2)xt-3 + sqrt(1-alphaT*alphaT-1*alphaT-2)*epsilon
and so on.

as a result we can use a cumulative alpha because we are using the same product cumulative products on both
finally we can write
alphaCumProd = alphaT * alphaT-1 * alphaT-2 * .... AlphaT-S


q(xt|x0) = Normal(sqrt(alphaCumProd)*x0, (1-alphaCumProd)I)
xt = sqrt(alphaCumProd)*x0 + sqrt(1-alphaCumpRod)*epsilon
'''

#Precompute all the noise values as a big vector
def forwardDiffusionValuesLinear(betaStart,betaEnd,T):

    betas = torch.linspace(betaStart,betaEnd,T)
    alpha = 1- betas
    alphaCumProd = torch.cumprod(alpha,dim=0)
    return betas,alpha,alphaCumProd

#use q(xt|x0) implementation
def qSample(xStart,t,alphaCumProd):
    epsilon = torch.randn_like(xStart)
    xT = torch.sqrt(alphaCumProd[t])*xStart + torch.sqrt(1-alphaCumProd[t]) *epsilon
    return xT

#consider Cosine forward Diffusion Later