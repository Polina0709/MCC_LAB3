import numpy as np
import matplotlib.pyplot as plt
import time

data = np.loadtxt('y9.txt').T

c1, c3, m2, m3 = 0.14, 0.2, 28, 18
t0, T, deltaT = 0, 50, 0.2
epsilon = 1e-5
c2, c4, m1 = 0.2, 0.1, 9

def SensMatrix(b):
    m1_inv, m3_inv, b2_inv = 1 / m1, 1 / m3, 1 / b[2] if b[2] != 0 else 0
    return np.array([
        [0, 1, 0, 0, 0, 0],
        [- (b[1] + b[0]) * m1_inv, 0, b[1] * m1_inv, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [b[1] * b2_inv, 0, -(b[1] + c3) * b2_inv, 0, c3 * b2_inv, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, c3 * m3_inv, 0, -(c4 + c3) * m3_inv, 0]
    ])

def ModelDerivatives(y, b):
    db0 = np.zeros((6, 6))
    db1 = np.zeros((6, 6))
    db2 = np.zeros((6, 6))

    db0[1, 0] = -1 / m1
    db1[1, 0] = -1 / m1
    db1[1, 2] = 1 / m1
    db2[3, 0] = -b[1] / (b[2] ** 2)
    db2[3, 2] = (b[1] + c3) / (b[2] ** 2)
    db2[3, 4] = -c3 / (b[2] ** 2)

    db0 = np.dot(db0, y)
    db1 = np.dot(db1, y)
    db2 = np.dot(db2, y)

    return np.array([db0, db1, db2]).T

def Sensitivity_RK(A, db, uu, deltaT, timeStamps):
    for i in range(1, len(timeStamps)):
        k1 = deltaT * (np.dot(A, uu[i - 1]) + db[i - 1])
        k2 = deltaT * (np.dot(A, (uu[i - 1] + k1 / 2)) + db[i - 1])
        k3 = deltaT * (np.dot(A, (uu[i - 1] + k2 / 2)) + db[i - 1])
        k4 = deltaT * (np.dot(A, (uu[i - 1] + k3)) + db[i - 1])

        uu[i] = uu[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return uu

def Model_RK(b, timeStamps, deltaT):
    yy = np.zeros_like(data)
    yy[0] = data[0].copy()
    A = SensMatrix(b)

    for i in range(1, len(timeStamps)):
        y_prev = yy[i - 1]
        k1 = deltaT * np.dot(A, y_prev)
        k2 = deltaT * np.dot(A, (y_prev + k1 / 2))
        k3 = deltaT * np.dot(A, (y_prev + k2 / 2))
        k4 = deltaT * np.dot(A, (y_prev + k3))
        yy[i] = y_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return yy

def DeltaB(uu, db, deltaT, timeStamps, data, b):
    diff_y = data - Model_RK(b, timeStamps, deltaT)

    du = (np.array([u.T @ u for u in uu]) * deltaT).sum(0)
    du_inv = np.linalg.inv(du)
    uY = (np.array([uu[i].T @ diff_y[i] for i in range(len(timeStamps))]) * deltaT).sum(0)
    deltaB = du_inv @ uY

    return deltaB

def Parameters(b, t0, T, deltaT, eps, max_iter=1000):
    timeStamps = np.linspace(t0, T, int((T - t0) / deltaT + 1))

    iteration_count = 0
    history = []
    prev_b = b.copy()
    diff_y_history = []

    c2_history = []
    c4_history = []
    m1_history = []

    while iteration_count < max_iter:
        iteration_count += 1

        yy = Model_RK(b, timeStamps, deltaT)

        uu = np.zeros((len(timeStamps), 6, 3))
        db = ModelDerivatives(yy.T, b)
        A = SensMatrix(b)

        uu = Sensitivity_RK(A, db, uu, deltaT, timeStamps)
        deltaB = DeltaB(uu, db, deltaT, timeStamps, data, b)

        b += deltaB

        history.append(b.copy())

        c2_history.append(b[0])
        c4_history.append(b[1])
        m1_history.append(b[2])

        if np.abs(deltaB).max() < eps:
            print(f"Convergence reached at iteration {iteration_count}")
            break

        if np.allclose(b, prev_b, atol=epsilon):
            break

        prev_b = b.copy()

    # Побудова графіка зміни параметрів
    plt.figure(figsize=(10, 6))

    plt.plot(range(iteration_count), c2_history, label='c2', color='blue')
    plt.plot(range(iteration_count), c4_history, label='c4', color='orange')
    plt.plot(range(iteration_count), m1_history, label='m1', color='green')

    plt.xlabel('Ітерація')
    plt.ylabel('Значення параметру')
    plt.title('Зміна патаметрів протягом ітерацій')
    plt.legend()
    plt.grid(True)

    plt.show()

    np.savetxt("parameters_history.txt", np.column_stack((c2_history, c4_history, m1_history)))

    return b, iteration_count, history, diff_y_history

if __name__ == "__main__":
    start_time = time.time()
    solution, iteration_count, history, diff_y_history = Parameters(np.array([c2, c4, m1]), t0, T, deltaT, epsilon)
    end_time = time.time()
    execution_time = end_time - start_time

    solution = np.round(solution, 4)

    print(f"Результати: c2 = {solution[0]}, c4 = {solution[1]}, m1 = {solution[2]}")
    print("Кількість ітерацій:", iteration_count)

    np.savetxt("final_parameters.txt", [solution], delimiter=",", header="c2,c4,m1", comments="")
