import timeit
import numpy as np
import matplotlib.pyplot as plt

def fast_fourier_transform(x):
    n = len(x)
    levels = int(np.log2(n))
    # Завдання початкових значень
    X = x.astype(complex)
    for level in range(levels):
        # Розбиття вектора на дві частини
        step = 2 ** level
        for k in range(step):
            # Обчислення W-матриці
            W = np.exp(-2j * np.pi * k / (2 * step))
            # Обчислення ШПФ для кожного з піввекторів та їх об'єднання
            for i in range(k, n, 2 * step):
                t = W * X[i + step]
                X[i + step] = X[i] - t
                X[i] += t
    return X

# Генеруємо випадковий сигнал довжиною 26
N = 512
x = np.random.rand(N)

# Доповнюємо вхідний сигнал нулями до степеня 2
M = 2**int(np.ceil(np.log2(N)))
x = np.concatenate([x, np.zeros(M-N)])

# Обчислюємо ШПФ за допомогою рекурсивного алгоритму
t1 = timeit.default_timer()
X = fast_fourier_transform(x)
t2 = timeit.default_timer()

# Виводимо результати
for i, val in enumerate(X):
    print(f"C_{i}: {val}")

# вивід часу обчислення
print(f"\nЧас виконання: {t2 - t1:7f} секунд")

# обрахунок кількості операцій
num_plus = N
num_mult = 4 * N
num_operations = num_plus + num_mult
print(f"Кількість операцій додавання: {num_plus}")
print(f"Кількість операцій множення: {num_mult}")
print(f"Кількість операцій: {num_operations}")

# обчислення спектру амплітуд і фаз для вхідного вектору
amp = np.abs(X[1:])
phase = np.angle(X)

# побудова графіку спектру амплітуд і фаз
plt.figure()
plt.stem(amp)
plt.title("Амплітудний спектр")
plt.xlabel("Частота")
plt.ylabel("Амплітуда")
plt.figure()
plt.stem(phase)
plt.title("Фазовий спектр")
plt.xlabel("Частота")
plt.ylabel("Фаза")
plt.show()