import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# --- Red de Hamming ---
class HammingNetwork:
    def __init__(self, patrones):
        self.p = np.array(patrones); self.np = len(patrones); self.lp = len(patrones[0]); self.w = self.p / 2; self.b = np.sum(self.p, axis=1) / 2
    def activar(self, entrada, it=5):
        s = np.zeros(self.np);
        for _ in range(it): s_prev = np.copy(s); s = np.maximum(0, self.b + np.dot(entrada, self.w.T) - (s - 1) / self.lp); if np.all(s == s_prev): break
        return s
    def reconocer(self, entrada, it=5): return self.p[np.argmax(self.activar(entrada, it))]

# --- Red de Hopfield ---
class HopfieldNetwork:
    def __init__(self, n): self.n = n; self.w = np.zeros((n, n))
    def aprender(self, patron): p = np.array(patron); self.w += np.outer(p, p); np.fill_diagonal(self.w, 0)
    def recordar(self, inicio, it=10): s = np.array(inicio); for _ in range(it): s_prev = np.copy(s); for i in range(self.n): s[i] = 1 if np.dot(s, self.w[:, i]) > 0 else -1; if np.all(s == s_prev): break; return s

# --- Aprendizaje Hebbiano ---
def hebb(entrada, salida, lr=0.1, pesos=None): e = np.array(entrada); w = np.zeros(len(e)) if pesos is None else np.array(pesos); return w + lr * salida * e

# --- Máquina de Boltzmann ---
class BoltzmannMachine:
    def __init__(self, n, ti=1.0, fc=0.95): self.n = n; self.w = np.zeros((n, n)); np.fill_diagonal(self.w, 0); self.t = ti; self.fc = fc
    def _p_act(self, estado, i): return 1 / (1 + np.exp(-np.dot(estado, self.w[:, i]) / self.t))
    def aprender(self, pos, neg, it=1000): pos = np.array(pos); neg = np.array(neg); for _ in range(it): for i in range(self.n): ep = self._p_act(pos, i); en = self._p_act(neg, i); for j in range(self.n): if i != j: self.w[i, j] += (ep * pos[j] - en * neg[j]); self.t *= self.fc; if self.t < 0.1: break
    def recordar(self, inicio, it=500): s = np.array(inicio); for _ in range(it): i = np.random.randint(self.n); s[i] = 1 if np.random.rand() < self._p_act(s, i) else 0; return s

if __name__ == "__main__":
    print("Hamming:"); ph = [[1, 1, 1, 1], [-1, -1, -1, -1]]; hn = HammingNetwork(ph); ih = [0.8, 0.9, 1.1, 0.7]; rh = hn.reconocer(ih); print(f"Entrada: {ih}, Reconocido: {rh}")
    print("\nHopfield:"); nh = 3; hf = HopfieldNetwork(nh); p_hf = [1, -1, 1]; hf.aprender(p_hf); si_hf = [1, 1, 1]; rr_hf = hf.recordar(si_hf); print(f"Aprendido: {p_hf}, Inicio: {si_hf}, Recordado: {rr_hf}")
    print("\nHebb:"); eh = [1, 0, 1, 1]; sh = 1; wh = hebb(eh, sh); print(f"Entrada: {eh}, Salida: {sh}, Pesos: {wh}")
    print("\nBoltzmann:"); nb = 2; bm = BoltzmannMachine(nb); pb = [1, 1]; nb_bm = [0, 0]; bm.aprender(pb, nb_bm, it=2000); sibm = [0, 1]; rbm = bm.recordar(sibm); print(f"Positivo: {pb}, Negativo: {nb_bm}, Inicio: {sibm}, Recordado: {rbm}")
