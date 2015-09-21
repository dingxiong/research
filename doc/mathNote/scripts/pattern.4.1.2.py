import sympy as sp

sp.init_printing()
ap, am = sp.symbols('a_+, a_-')
print sp.Integral(1/ap, ap)
