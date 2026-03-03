import numpy as np


# ===============================
# basic utils
# ===============================

def convmtx(x, L):

    N = len(x)

    X = np.zeros((N, L))

    for i in range(L):

        X[i:, i] = x[:N-i]

    return X


def fir(x, h):

    return convmtx(x, len(h)) @ h


def poly(x, a):

    return (
        a[0]*x +
        a[1]*x**3 +
        a[2]*x**5
    )


def nmse(x, y):

    return 10*np.log10(np.mean((x-y)**2)/np.mean(y**2))


# ===============================
# true PA system
# ===============================

def pa(x):

    h1 = np.array([0.9, 0.3, -0.1])

    a = np.array([1.0, 0.4, 0.1])

    h2 = np.array([1.0, -0.2, 0.05])

    v = fir(x, h1)

    w = poly(v, a)

    y = fir(w, h2)

    return y


# ===============================
# training WH inverse
# ===============================

def train_wh_inverse(x, y, L=5, iters=10):

    g1 = np.random.randn(L)*0.01

    g2 = np.random.randn(L)*0.01

    a = np.array([1.0, 0.0, 0.0])


    for i in range(iters):

        # update g1

        Y = convmtx(y, L)

        u = Y @ g1

        Phi = np.column_stack([u, u**3, u**5])

        V = convmtx(Phi @ a, L)

        g2 = np.linalg.lstsq(V, x, rcond=None)[0]


        # update a

        u = fir(y, g1)

        Phi = np.column_stack([u, u**3, u**5])

        z = fir(Phi @ a, g2)

        a = np.linalg.lstsq(Phi, x, rcond=None)[0]


        # update g1

        U = convmtx(y, L)

        g1 = np.linalg.lstsq(U, x, rcond=None)[0]


        # evaluate

        x_hat = fir(poly(fir(y, g1), a), g2)

        print(f"iter {i+1} NMSE:", nmse(x_hat, x))


    return g1, a, g2


# ===============================
# main
# ===============================

N = 20000

x = np.random.randn(N)

y = pa(x)


g1, a, g2 = train_wh_inverse(x, y)


x_hat = fir(poly(fir(y, g1), a), g2)


print("FINAL NMSE:", nmse(x_hat, x))
