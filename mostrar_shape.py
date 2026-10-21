"""Abre um arquivo da base e mostra o shape dos dados.
Só precisa de numpy e scipy (não usa o torch).

Uso:
  python mostrar_shape.py            # abre dados/s1.mat
  python mostrar_shape.py 5          # abre dados/s5.mat
"""
import sys
import numpy as np
import scipy.io as sio
from scipy.signal import cheby1, filtfilt

FS = 256
TECLAS = list(" adeilmnorsu")

s = int(sys.argv[1]) if len(sys.argv) > 1 else 1
arq = f"dados/s{s}.mat"

mat = sio.loadmat(arq)
eeg = mat["eeg"]
print(f"arquivo: {arq}")
print(f"variáveis no .mat: {[k for k in mat if not k.startswith('__')]}")
print(f"shape de eeg: {eeg.shape}   tipo: {eeg.dtype}")
print("  eixo 0 = teclas      :", eeg.shape[0])
print("  eixo 1 = canais      :", eeg.shape[1])
print("  eixo 2 = amostras    :", eeg.shape[2], f"({eeg.shape[2] / FS:.2f} s a {FS} Hz)")
print("  eixo 3 = tentativas  :", eeg.shape[3])

k = TECLAS.index("o")
print("\nexemplo: tecla 'o' (14,25 Hz), canal 1, tentativa 1, primeiras 5 amostras (valor bruto, com offset do aparelho):")
print(" ", eeg[k, 0, :5, 0].round(2))
b, a = cheby1(4, 0.5, [6 / (FS / 2), 80 / (FS / 2)], btype="band")
print("  depois do filtro 6-80 Hz (offset removido):", filtfilt(b, a, eeg[k, :, :, 0], axis=-1)[0, 100:105].round(2))

# mesma reorganização que ssvep.carregar()
X = eeg.transpose(3, 0, 1, 2).reshape(-1, eeg.shape[1], eeg.shape[2])
y = np.tile(np.arange(eeg.shape[0]), eeg.shape[3])
print("\ndepois de reorganizar (carregar):")
print(f"  X: {X.shape}  -> (tentativas, canais, amostras)")
print(f"  y: {y.shape}       -> tecla de cada tentativa, ex.: {y[:12]}")
