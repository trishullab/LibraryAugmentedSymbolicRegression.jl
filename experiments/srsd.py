# SRSD Dataset @ https://arxiv.org/pdf/2206.10540.pdf 
import numpy as np

pi = np.pi
cos = np.cos
sin = np.sin
sqrt = np.sqrt

# for really small values (e.g. 1e-24), change units to scale to reasonable value for precision purposes
# (will say divide by 1e-24 for example, really means divided true value by that)
# add context to this dataset

dataset_easy = [
            ("F = mu * N", {"F": None,"mu": ("loguniform", (0.01, 1)),"N": ("loguniform", (0.01,1))}),
            ("E = q / (4 * pi * e * r^2)", {"E": None,"q": ("loguniform", (0.1, 10)),"r": ("loguniform", (0.1,10)),"e": ("constant", 8.854e-12)}),
            ("F = q * E", {"F": None,"q": ("loguniform", (0.1, 10)),"E": ("loguniform", (0.1,10))}),
            ("U = m * g * z", {"U": None,"m": ("loguniform", (0.01, 1)),"z": ("loguniform", (0.01,1)), "g": ("constant", 9.807)}),
            ("U = k * x^2 / 2", {"U": None,"k": ("loguniform", (100, 10000)),"x": ("loguniform", (0.01,1))}),
            ("T = r * F * sin(theta)", {"T": None,"r": ("loguniform", (0.1, 10)),"F": ("loguniform", (0.1,10)), "theta": ("uniform", (0, 2 * pi))}),
            ("L = m * r * v * sin(theta)", {"L": None,"m": ("loguniform", (0.1, 10)),"r": ("loguniform", (0.1, 10)),"v": ("loguniform", (0.1,10)), "theta": ("uniform", (0, 2 * pi))}),
            ("V = q / C", {"V": None,"q": ("loguniform", (1e-5, 1e-3)),"C": ("loguniform", (1e-5, 1e-3))}),
            ("n = sin(theta1) / theta2", {"n": None,"theta1": ("uniform", (0, pi / 2)),"theta2": ("uniform", (0, pi/2))}),
            ("f = 1 / ((1/d1) + (n/d2))", {"f": None,"d1": ("loguniform", (1e-3, 0.1)),"d2": ("loguniform", (1e-3, 0.1)),"n": ("loguniform", (0.1, 10))}),
            ("d = lam / (n * sin(theta))", {"d": None,"lam": ("loguniform", (1e+1, 1e+3)),"n": ("loguniform", (1, 100)),"theta": ("uniform", (0, pi/2))}), # divide lam by 1e-12
            ("v = mu * q * V / d", {"v": None,"mu": ("loguniform", (1e0, 1e+2)),"q": ("loguniform", (1e+1, 1e+3)),"V": ("loguniform", (0.1, 10)),"d": ("loguniform", (1e-3, 0.1))}), # divide q by 1e-12 and mu by 1e-6
            ("c = sqrt(gam * P / p)", {"c": None,"gam": ("uniform", (1,2)),"P": ("uniform", (5e0, 1.5e+1)),"p": ("uniform", (1,2))}), # divide P by 1e-6
            ("J = k * (T2 - T1) * A / d", {"J": None,"k": ("loguniform", (0.1,10)),"T2": ("loguniform", (10, 100)),"T1": ("loguniform", (10, 100)),"A": ("loguniform", (1e-4,1e-2)),"d": ("loguniform", (1e-2,1))}),
            ("h = W / (4 * pi * r^2)", {"h": None,"W": ("loguniform", (1,100)),"r": ("loguniform", (0.01, 1))}),
            ("phi = q / (4 * pi * e * r)", {"phi": None,"q": ("loguniform", (1e-3, 0.1)),"r": ("loguniform", (0.01,1)),"e": ("constant", 8.854e-12)}),
            ("u = e * E^2 / 2", {"u": None,"e": ("constant", 8.854),"E": ("loguniform", (10,1e3))}), # divide e by 1e-12
            ("E = sig / (e * (1 + chi))", {"E": None,"e": ("constant", 8.854e-12),"sig": ("loguniform", (1e-3,0.1)),"chi": ("loguniform", (1,100))}),
            ("B = 2 * I / (4 * pi * e * r * c^2)", {"B": None,"e": ("constant", 8.854e-12),"I": ("loguniform", (1e-3,0.1)),"r": ("loguniform", (1e-3,0.1)),"c": ("constant", 2.998)}), # divide c by 1e-8
            ("U = - mu * B * cos(theta)", {"U": None,"mu": ("loguniform", (0.1,10)),"B": ("loguniform", (1e-3,0.1)),"theta": ("uniform", (0, 2*pi))}), # divide mu by 1e-24 for real bounds
            ("U = - p * E * cos(theta)", {"U": None,"p": ("loguniform", (1e2,1e4)),"E": ("loguniform", (10,1e3)),"theta": ("uniform", (0, 2*pi))}), # divide mu by 1e-24 for real bounds
            ("L = e * c * E^2", {"L": None,"e": ("constant", 8.854e-12),"E": ("loguniform", (0.1,10)),"c": ("constant", 2.998e8)}),
            ("u = e * E^2", {"u": None,"e": ("constant", 8.854),"E": ("loguniform", (0.1,10))}), # divide e by 1e-12
            ("w = g * q * B / (2 * m)", {"u": None,"g": ("uniform", (-1,1)),"q": ("loguniform", (1e-3,1e-1)),"B": ("loguniform", (1e-1,1e+1)),"m": ("loguniform", (1e-6,1e-4))}), # divide m by 1e-24 for real bounds, q and B by 1e-8
            ("U = 2 * pi * g * mu * B * Jz / h", {"U": None,"g": ("uniform", (-1,1)),"mu": ("constant", 9.2740100783),"B": ("loguniform", (1e-3,0.1)),"Jz": ("loguniform", (1e-2,1e2)),"h": ("constant", 6.626e-10)}), # divide mu, Jz, h by 1e-24 for real bounds
            ("F = Y * A * Dl / l", {"F": None,"Y": ("loguniform", (0.1,10)),"A": ("loguniform", (1e-4,1e-2)),"Dl": ("loguniform", (1e-3,1e-1)),"l": ("loguniform", (1e-2,1))}),
            ("mu = Y / (2 * (1 + sig))", {"mu": None,"Y": ("loguniform", (0.1,10)),"sig": ("loguniform", (1e-2,1))}),
            ("w = 4 * pi * mu * B / h", {"w": None,"mu": ("loguniform", (1e-11,1e-9)),"B": ("loguniform", (1e-3,1e-1)),"h": ("constant", 6.6261e-10)}), # divide h by 1e-24 for real bounds
            ("J = m * h / (2 * pi)", {"J": None,"m": ("loguniform", (1,100)),"h": ("constant", 6.6261)}), # divide h by 1e-34 for real bounds
            ("k = 2 * pi * s / (N * b)", {"k": None,"s": ("loguniform", (1,100)),"N": ("loguniform", (1,100)),"b": ("loguniform", (1e-10,1e-8))}),
        ]

hints_easy = [
            "Simple relationship.",
            "One variable is a radius and the square of its inverse is useful.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship and the square of a variable is useful.",
            "Simple relationship and trigonimetric properties are used.",
            "Simple relationship and trigonimetric properties are used.",
            "Simple relationship.",
            "Simple relationship and trigonimetric properties are used.",
            "Reciprocals of variables are used.",
            "Reciprocals and trigonimetric properties are used.",
            "Simple relationship.",
            "Square root is useful.",
            "Simple relationship.",
            "One variable is a radius and the square of its inverse is useful.",
            "One variable is a radius and the its inverse is useful.",
            "Simple relationship and the square of a variable is useful.",
            "Simple relationship with a reciprocal.",
            "Simple relationship with a reciprocal.",
            "Simple relationship and trigonimetric properties are used.",
            "Simple relationship and trigonimetric properties are used.",
            "Simple relationship and the square of a variable is useful.",
            "Simple relationship and the square of a variable is useful.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            ]


dataset_medium = [
            ("d = sqrt((x2 - x1)^2 + (y2 - y1)^2)", {"d": None,"x1": ("loguniform", (0.1, 10)),"x2": ("loguniform", (0.1, 10)),"y1": ("loguniform", (0.1, 10)),"y2": ("loguniform", (0.1, 10))}),
            ("m = m0 / sqrt(1 - (v^2 / c^2))", {"m": None,"m0": ("loguniform", (0.1, 10)),"v": ("loguniform", (1e5, 1e8)),"c": ("constant", 2.998e8)}),
            ("A = x1 * y1 + x2 * y2 + x3 * y3", {"A": None,"x1": ("loguniform", (0.1, 10)),"x2": ("loguniform", (0.1, 10)),"x3": ("loguniform", (0.1, 10)),"y1": ("loguniform", (0.1, 10)),"y2": ("loguniform", (0.1, 10)),"y3": ("loguniform", (0.1, 10))}),
            ("F = q1 * q2 / (4 * pi * e * r^2)", {"F": None,"q1": ("loguniform", (0.1, 10)),"q2": ("loguniform", (0.1, 10)),"r": ("loguniform", (0.1, 10)),"e": ("constant", 8.854e-12)}),
            ("F = q * (E + B * v * sin(theta))", {"F": None,"q": ("loguniform", (0.1, 10)),"E": ("loguniform", (0.1, 10)),"B": ("loguniform", (0.1, 10)),"v": ("loguniform", (0.1, 10)),"theta": ("uniform", (0,pi/2))}),
            ("K = (m * (v^2 + u^2 + w^2)) / 2", {"K": None,"m": ("loguniform", (0.1, 10)),"v": ("loguniform", (0.1, 10)),"u": ("loguniform", (0.1, 10)),"w": ("loguniform", (0.1, 10))}),
            ("U = G * m1 * m2 * ((1/r2) - (1/r1))", {"U": None,"G": ("constant", 6.674),"m1": ("loguniform", (0.01, 1)),"m2": ("loguniform", (0.01, 1)),"r1": ("loguniform", (0.01, 1)),"r2": ("loguniform", (0.01, 1))}), # divide G by 1e-12 for real bounds
            ("p = (m0 * v) / sqrt(1 - (v^2 / c^2))", {"m": None,"m0": ("loguniform", (0.01, 1)),"v": ("loguniform", (1e5, 1e7)),"c": ("constant", 2.998e8)}),
            ("v1 = (u + v) / (1 + (u * v / c^2))", {"v1": None,"u": ("loguniform", (1e6, 1e8)),"v": ("loguniform", (1e6, 1e8)),"c": ("constant", 2.998e8)}),
            ("r = (m1 * r1 + m2 * r2) / (m1 + m2)", {"r": None,"m1": ("loguniform", (0.1, 10)),"m2": ("loguniform", (0.1, 10)),"r1": ("loguniform", (0.1, 10)),"r2": ("loguniform", (0.1, 10))}),
            ("E = (m * (w^2 + w0^2) * x^2) / 4", {"E": None,"m": ("loguniform", (0.1, 10)),"w": ("loguniform", (0.1, 10)),"w0": ("loguniform", (0.1, 10)),"x": ("loguniform", (0.1, 10))}),
            ("k = w / c", {"k": None,"w": ("loguniform", (1e9, 1e11)),"c": ("constant", 2.998e8)}),
            ("P = (q^2 * a^2) / (6 * pi * e * c^3)", {"P": None,"q": ("loguniform", (1e-3, 1e-1)),"a": ("loguniform", (1e5, 1e7)),"e": ("constant", 8.854e-12),"c": ("constant", 2.998e8)}),
            ("w = (q * v * B) / p", {"w": None,"q": ("loguniform", (1e-11, 1e-9)),"v": ("loguniform", (1e5, 1e7)),"B": ("loguniform", (1e1, 1e3)),"p": ("loguniform", (1e9, 1e11))}),
            ("w = w0 / (1 - (v/c))", {"w": None,"w0": ("loguniform", (1e9, 1e11)),"v": ("loguniform", (1e5, 1e7)),"c": ("constant", 2.998e8)}),
            ("W = (h * w) / (2 * pi)", {"W": None,"h": ("constant", 6.626e-10),"w": ("loguniform", (1e9, 1e11))}), # divide h by 1e-24 for real bounds
            ("r = 4 * pi * e * (h / (2 * pi))^2 / (m * q^2)", {"r": None,"e": ("constant", 8.854),"h": ("constant", 6.626e-10),"m": ("loguniform", (1e-10, 1e-8)),"q": ("loguniform", (1e-1, 1e+1))}), # divide h by 1e-24, m by 1e-18, and q by 1e-10 for real bounds
            ("U = 3 * P * V / 2", {"U": None,"P": ("loguniform", (1e4, 1e6)),"V": ("loguniform", (1e-5, 1e-3))}),
            ("U = P * V / (gam - 1)", {"U": None,"P": ("loguniform", (1e4, 1e6)),"V": ("loguniform", (1e-5, 1e-3)),"gam": ("uniform", (1, 2))}),
            ("D = mu * k * T", {"D": None,"mu": ("loguniform", (1e13, 1e15)),"k": ("constant", 1.381e1),"T": ("loguniform", (1e1, 1e3))}), # divide k by 1e-24 for real bounds
            ("K = k * v / (sig * (gam - 1))", {"K": None,"k": ("constant", 1.381e1),"v": ("loguniform", (1e2, 1e4)),"sig": ("loguniform", (1e3, 1e5)),"gam": ("uniform", (1, 2))}), # divide k, sig by 1e-24 for real bounds  
            ]

hints_medium = [
            "Big square root.",
            "Big square root reciprocal.",
            "Simple relationship.",
            "One variable is a radius and the square of its inverse is useful.",
            "Simple relationship and trigonimetric properties are used.",
            "Simple relationship of quadratic nature.",
            "Two variables are radii and the their inverse is useful.",
            "Big square root reciprocal.",
            "Simple relationship of inverse quadratic nature.",
            "Simple relationship.",
            "Simple relationship of quadratic nature.",
            "Simple relationship.",
            "Simple relationship of quadratic and cubic nature.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            "Simple relationship.",
            ]

dataset_hard = []

hints_hard = []