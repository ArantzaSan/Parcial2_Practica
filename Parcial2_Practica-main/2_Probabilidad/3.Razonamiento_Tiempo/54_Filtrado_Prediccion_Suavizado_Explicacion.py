import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def media_movil(data, w):
    return np.convolve(data, np.ones(w) / w, mode='same')

def gaussiano(data, sigma):
    w = int(4 * sigma + 1)
    g = gaussian(w if w % 2 else w + 1, sigma)
    return convolve(data, g / np.sum(g), mode='same')

def naive(data, steps=1):
    return np.array([data[-1]] * steps)

def exp_simple(data, alpha=0.3, steps=5):
    s = np.zeros_like(data, dtype=float)
    s[0] = data[0]
    for i in range(1, len(data)):
        s[i] = alpha * data[i] + (1 - alpha) * s[i-1]
    return s, np.array([s[-1]] * steps)

def exp_doble(data, alpha=0.3, beta=0.2, steps=5):
    n = len(data)
    l, t, f = np.zeros(n), np.zeros(n), np.zeros(n)
    l[0] = data[0]
    t[0] = data[1] - data[0] if n > 1 else 0
    f[0] = l[0]
    for i in range(1, n):
        l[i] = alpha * data[i] + (1 - alpha) * (l[i-1] + t[i-1])
        t[i] = beta * (l[i] - l[i-1]) + (1 - beta) * t[i-1]
        f[i] = l[i]
    return f, np.array([l[n-1] + (i + 1) * t[n-1] for i in range(steps)])

def exp_triple(data, steps=10):
    try:
        m = ExponentialSmoothing(data, seasonal='add', seasonal_periods=7, initialization_method="estimated").fit()
        return m.fittedvalues, m.forecast(steps)
    except:
        return np.full_like(data, np.nan), np.full(steps, np.nan)

def plot(o, f=None, s=None, p=None, fm=None, pm=None, ps=None):
    plt.plot(o, label='Original', alpha=0.7)
    if f is not None: plt.plot(f, label=f'Filtrado ({fm})', linestyle='--')
    if s is not None: plt.plot(s, label=f'Suavizado ({pm})', linestyle='--')
    if p is not None:
        xp = np.arange(len(o), len(o) + ps)
        plt.plot(xp, p, label=f'Predicción ({pm})', marker='o')
    plt.title('Filtrado, Suavizado y Predicción')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    n = 100
    t = np.linspace(10, 30, n)
    r = np.random.normal(0, 5, n)
    o = t + r

    w = 5; f_mm = media_movil(o, w); plot(o, f=f_mm, fm="MM")
    sg = 2; f_g = gaussiano(o, sg); plot(o, f=f_g, fm="Gaussiano")
    ps_n = 5; p_n = naive(o, ps_n); plot(o, p=p_n, pm="Naive", ps=ps_n)
    a_es = 0.3; s_es, p_es = exp_simple(o, a_es, 5); plot(o, s=s_es, p=p_es, pm="ES Simple", ps=5)
    a_ed = 0.5; b_ed = 0.2; s_ed, p_ed = exp_doble(o, a_ed, b_ed, 5); plot(o, s=s_ed, p=p_ed, pm="ES Doble", ps=5)
    s_et, p_et = exp_triple(o, 10); plot(o, s=s_et, p=p_et, pm="ES Triple", ps=10)
