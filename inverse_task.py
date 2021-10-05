import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.interpolate as spi

sigma = 0.5
t_init = 0
t_end = 1
x_end = 1
x_init = -1
Nx = 600  # кол-во узлов по пространству
Nt = 300  # кол-во узлов по времени
n = 500  # кол-во данных
Number_iter = 800  # кол-во итераций
T_fix = 300  # номер узла(!) фиксации
delta = 0.15  # уровень шума

def dW(delta_t):
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

# Метод Эйлера-Маруяма для решения СДУ для точного мю
def EM(sigma, dt, time, N, seed):
    np.random.seed(seed)
    X_0 = 0
    mu = np.zeros(N + 1)
    X_new = np.zeros(N + 1)
    X_new[0] = X_0
    mu[0] = -5 * X_0 ** 3 + 2 * X_0  # точное мю
    for i in range(1, time.size):
        y = X_new[i - 1]
        X_new[i] = y + mu[i - 1] * dt + sigma * dW(dt)
        mu[i] = -5 * X_new[i] ** 3 + 2 * X_new[i]
    return X_new

def Empirical_density(X, X_mesh_for_interp):
    N = len(X)
    num_density = 1 + int(math.log2(N))
    Max = max(X)
    Min = min(X)
    h_step = (Max - Min) / (num_density - 1)
    dens_step = [Min]
    Uo = np.zeros(num_density)
    X_mesh = np.zeros(num_density)
    for i in range(0, num_density):
        ai = Min + i * h_step
        ai1 = ai + h_step
        count = 0
        for j in range(0, N):
            if (ai <= X[j] < ai1):
                count = count + 1
        Uo[i] = count / (h_step * (N + 1))
        dens_step.append(ai1)
        X_mesh[i] = (h_step) / 2 + Min + i * h_step

    U = np.zeros(len(Uo) + 2)
    X_mesh_for_U = np.zeros(len(X_mesh) + 2)
    for i in range(0, num_density):
        U[i + 1] = Uo[i]
        X_mesh_for_U[i + 1] = X_mesh[i]
    X_mesh_for_U[0] = X_mesh_for_interp[0]
    X_mesh_for_U[len(X_mesh) + 1] = X_mesh_for_interp[len(X_mesh_for_interp) - 1]

    Uo_new = spi.interp1d(X_mesh_for_U, U, kind='cubic', fill_value="extrapolate")(X_mesh_for_interp)

    return Uo_new

# Нахождение решения ур-ния Фоккера-Планка. Неявная схема. Прогонка.
def Fokker_Plank_schem1_for_mu(mu, sigma, X, T):

    tau = T[1] - T[0]
    U = np.zeros((len(X), len(T)))
    Nx = len(X)
    A = np.zeros(Nx)
    B = np.zeros(Nx)
    C = np.zeros(Nx)
    F = np.zeros(Nx)

    z = 4

    # граничные условия
    for i in range(0, len(T)):
        U[0][i] = 0
        U[len(X) - 1][i] = 0

    # начальные условия
    for i in range(0, Nx):
        U[i][0] = (z / (np.sqrt(math.pi))) * np.exp(-(z * X[i]) ** 2)

    for n in range(1, len(T)):

        for i in range(0, Nx - 1):
            A[i] = (-mu[i] / (X[i + 1] - X[i - 1])) - ((sigma ** 2) / (2 * (X[i + 1] - X[i]) * (X[i] - X[i - 1])))
            B[i] = (1 / tau) + ((mu[i + 1] - mu[i - 1]) / (X[i + 1] - X[i - 1])) + (
                    (sigma ** 2) / ((X[i + 1] - X[i]) * (X[i] - X[i - 1])))
            C[i] = (mu[i] / (X[i + 1] - X[i - 1])) - ((sigma ** 2) / (2 * (X[i + 1] - X[i]) * (X[i] - X[i - 1])))

        for i in range(1, Nx - 2):
            F[i] = (1 / tau) * U[i][n - 1]

        F[0] = 0  # из граничных условий
        F[len(X) - 2] = 0  # из граничных условий

        alpha = [-C[0] / B[0]]
        beta = [F[0] / B[0]]

        for i in range(1, Nx - 1):
            alpha.append((-C[i]) / (A[i] * alpha[i - 1] + B[i]))
            beta.append((F[i] - A[i] * beta[i - 1]) / (A[i] * alpha[i - 1] + B[i]))

        U[len(X) - 2][n] = beta[Nx - 2]

        for i in range(Nx - 2, 1, -1):
            U[i - 1][n] = alpha[i - 1] * U[i][n] + beta[i - 1]
    Result = U.transpose()

    return Result

# нахождение F'[mu]
def derivative_Fokker_Plank_schem1_for_mu(mu, sigma, X, T, U):
    tau = T[1] - T[0]
    V = np.zeros((len(X), len(T)))
    Nx = len(X)
    A = np.zeros(Nx)
    B = np.zeros(Nx)
    C = np.zeros(Nx)
    F = np.zeros(Nx)

    # граничные условия
    for i in range(0, len(T)):
        V[0][i] = 0
        V[len(X) - 1][i] = 0

    # начальные условия
    for i in range(0, Nx):
        V[i][0] = 0

    for n in range(1, len(T)):

        for i in range(0, Nx - 1):
            A[i] = (-mu[i] / (X[i + 1] - X[i - 1])) - ((sigma ** 2) / (2 * (X[i + 1] - X[i]) * (X[i] - X[i - 1])))
            B[i] = (1 / tau) + ((mu[i + 1] - mu[i - 1]) / (X[i + 1] - X[i - 1])) + (
                    (sigma ** 2) / ((X[i + 1] - X[i]) * (X[i] - X[i - 1])))
            C[i] = (mu[i] / (X[i + 1] - X[i - 1])) - ((sigma ** 2) / (2 * (X[i + 1] - X[i]) * (X[i] - X[i - 1])))

        for i in range(1, Nx - 2):
            F[i] = (1 / tau) * V[i][n - 1] - (U[n][i + 1] - U[n][i - 1]) / (X[i + 1] - X[i - 1])

        F[0] = 0  # из граничных условий
        F[len(X) - 2] = 0  # из граничных условий

        alpha = [-C[0] / B[0]]
        beta = [F[0] / B[0]]

        for i in range(1, Nx - 1):
            alpha.append((-C[i]) / (A[i] * alpha[i - 1] + B[i]))
            beta.append((F[i] - A[i] * beta[i - 1]) / (A[i] * alpha[i - 1] + B[i]))

        V[len(X) - 2][n] = beta[Nx - 2]

        for i in range(Nx - 2, 1, -1):
            V[i - 1][n] = alpha[i - 1] * V[i][n] + beta[i - 1]
    Result = V.transpose()
    return Result

# нахождение оператора А^* = интегра от 0 до Т от функции W(t,x) dt
def adjoint(mu, sigma, X, T, g):
    tau = T[1] - T[0]
    W = np.zeros((len(X), len(T)))
    Nx = len(X)
    Nt = len(T)

    A = np.zeros(Nx)
    B = np.zeros(Nx)
    C = np.zeros(Nx)
    F = np.zeros(Nx)

    # граничные условия
    for i in range(0, len(T)):
        W[0][i] = 0
        W[len(X) - 1][i] = 0

    # начальные условия
    for i in range(0, Nx):
        W[i][Nt - 1] = g[i]

    for n in range(len(T) - 2, -1, -1):

        for i in range(0, Nx - 1):
            A[i] = (-mu[i] / (X[i + 1] - X[i - 1])) + ((sigma ** 2) / (2 * (X[i + 1] - X[i]) * (X[i] - X[i - 1])))
            B[i] = (-1 / tau) - ((sigma ** 2) / ((X[i + 1] - X[i]) * (X[i] - X[i - 1])))
            C[i] = (mu[i] / (X[i + 1] - X[i - 1])) + ((sigma ** 2) / (2 * (X[i + 1] - X[i]) * (X[i] - X[i - 1])))

        for i in range(1, Nx - 2):
            F[i] = (-1 / tau) * W[i][n + 1]

        F[0] = 0  # из граничных условий
        F[len(X) - 2] = 0  # из граничных условий

        alpha = [-C[0] / B[0]]
        beta = [F[0] / B[0]]

        for i in range(1, Nx - 1):
            alpha.append((-C[i]) / (A[i] * alpha[i - 1] + B[i]))
            beta.append((F[i] - A[i] * beta[i - 1]) / (A[i] * alpha[i - 1] + B[i]))

        W[len(X) - 2][n] = beta[Nx - 2]

        for i in range(Nx - 2, 1, -1):
            W[i - 1][n] = alpha[i - 1] * W[i][n] + beta[i - 1]

    Solve_adjoint = W.transpose()

    # интегрируем от 0 до Т найденное W(t,x)
    Int = np.zeros(Nx)
    for i in range(0, Nx):
        Int[i] = ((T[Nt - 1] - T[0]) / Nt) * ((Solve_adjoint[0][i] + Solve_adjoint[Nt - 1][i]) / 2)
        for j in range(1, Nt - 1):
            Int[i] = Int[i] + ((T[Nt - 1] - T[0]) / Nt) * Solve_adjoint[j][i]

    return Int


def Norma_L2(z, X):
    M = np.zeros(len(z))
    for i in range(0, len(z)):
        M[i] = abs(z[i]) ** 2

    Nx = len(X)

    Norma_L2 = ((X[Nx - 1] - X[0]) / Nx) * ((M[0] + M[Nx - 1]) / 2)
    for i in range(1, Nx - 1):
        Norma_L2 = Norma_L2 + ((X[Nx - 1] - X[0]) / Nx) * M[i]

    return Norma_L2

# массив для Х(t) из СДУ
X = []

# введем равномерную сетку по времни
dt = float(t_end - t_init) / Nt
time = np.arange(t_init, t_end + dt, dt)

# генерируем n путей для ТОЧНОГО мю
for j in range(0, n):
    seed = j + 2
    X.append(EM(sigma, dt, time, Nt, seed))

X_for_T_fix = []
for j in range(0, int(n)):
    X_for_T_fix.append(X[j][T_fix])

# упорядочим массив данных Х(Т)
X_for_T_fix.sort()

X_noise = np.zeros(len(X_for_T_fix))

for i in range(0, len(X_for_T_fix)):
    X_noise[i] = X_for_T_fix[i] + delta * max(X_for_T_fix) * np.random.normal()

# равномерная сетка по пространству
dx = float(x_end - x_init) / Nx
X_mesh = np.arange(x_init, x_end + dx, dx)

# эмпирическая функция распределения для данных
U_obs = Empirical_density(X_for_T_fix, X_mesh)

# точное значение мю
mu_exact = np.zeros(len(X_mesh))
for i in range(0, len(X_mesh)):
    mu_exact[i] = -5 * X_mesh[i] ** 3 + 2 * X_mesh[i]

# начальное приближение мю
mu = np.zeros(len(X_mesh))
mu_0 = np.zeros(len(X_mesh))
for i in range(0, len(X_mesh)):
    mu[i] = 2
    mu_0[i] = 2

J = np.zeros(Number_iter)
Eps_1 = np.zeros(Number_iter)
Eps_2 = np.zeros(Number_iter)
Eps_k = np.zeros(Number_iter)
Num = np.zeros(Number_iter)
# начало цикла по итерациям
for k in range(0, Number_iter):

    # U - решение уравнения КФП для мю_к
    U = Fokker_Plank_schem1_for_mu(mu, sigma, X_mesh, time)

    # решение уравнения Ф-П в фиксированный момент времени
    U_fix_time = np.zeros(len(X_mesh))

    G = np.zeros(len(X_mesh))
    #G_noise = np.zeros(len(X_mesh))

    Delta_mu = np.zeros(len(X_mesh))

    for i in range(0, len(X_mesh)):
        U_fix_time[i] = U[T_fix][i]
        G[i] = U_fix_time[i] - U_obs[i]
        Delta_mu[i] = mu_exact[i] - mu[i]

    Num[k] = k
    J[k] = Norma_L2(G, X_mesh)
    Eps_1[k] = Norma_L2(Delta_mu, X_mesh)
    Eps_2[k] = Norma_L2(mu_exact, X_mesh)
    Eps_k[k] = Eps_1[k] / Eps_2[k]
    # W = A^* G
    W = adjoint(mu, sigma, X_mesh, time, G)

    mu_next_iteration = np.zeros(len(X_mesh))

    # метод Ландвебера
    for i in range(0, len(X_mesh)):
        mu_next_iteration[i] = mu[i] - 0.5 * U_fix_time[i] * (W[i] - W[i - 1]) / (X_mesh[i] - X_mesh[i - 1])

    for i in range(0, len(X_mesh)):
        mu[i] = mu_next_iteration[i]

# ГРАФИКИ
mu_approx = np.zeros(len(X_mesh))
for i in range(0, len(X_mesh)):
    mu_approx[i] = mu[i]

L = [0, 100, 200, 300, 400, 500, 600, 700, 800]

plt.plot(Num, J, label='Значение функционала J')
plt.xlabel('Номер итерации', fontsize=16)
plt.ylabel('J', fontsize=16)
plt.xticks(L)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()

plt.plot(Num, Eps_k, label='относительная ошибка')
plt.xlabel('Номер итерации', fontsize=16)
plt.ylabel('Eps', fontsize=16)
plt.xticks(L)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()

plt.plot(X_mesh, mu_exact, label='mu точное')
plt.plot(X_mesh, mu_approx, label='mu на последней итерации', color='r')
plt.plot(X_mesh, mu_0, label=' mu первое приближение', color='g')
plt.xlabel('X', fontsize=16)
plt.ylabel('mu', fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()
