from scipy.integrate import quad

def ret(x):
  return x**2

def integrand(x, a, b):
  return a * ret(x) + b

a = 2
b = 1

I = quad(integrand, 0, 1, args=(a, b))
print(I)