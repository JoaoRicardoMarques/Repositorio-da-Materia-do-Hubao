"""Mostra os dados brutos de uma tentativa, o sinal filtrado e o espectro (FFT).
Gera a figura sinal_s<sujeito>_<letra>.png. Precisa de numpy, scipy e matplotlib (não usa o torch).

Uso:
  python mostrar_sinal.py            # sujeito 9 (de teste), tecla 'o' (14,25 Hz)
  python mostrar_sinal.py 9 a        # sujeito 9, tecla 'a' (11,25 Hz)
"""
import sys
import numpy as np
import scipy.io as sio
from scipy.signal import cheby1, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 256
TECLAS = list(" adeilmnorsu")
FREQS = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
INICIO = 39                       # amostra em que a tecla começa a piscar (0,15 s)
LAT = int(round(0.135 * FS))      # latência visual

s = int(sys.argv[1]) if len(sys.argv) > 1 else 9
letra = sys.argv[2] if len(sys.argv) > 2 else "o"
k = TECLAS.index(letra)
f0 = FREQS[k]

eeg = sio.loadmat(f"dados/s{s}.mat")["eeg"]          # (12, 8, 1114, 15)
b, a = cheby1(4, 0.5, [6 / (FS / 2), 80 / (FS / 2)], btype="band")
bruto = eeg[k, 0, :, 0]                               # canal 1, tentativa 1
filtrado_todos = filtfilt(b, a, eeg[k], axis=1)       # (8 canais, 1114, 15 tentativas)
filtrado = filtrado_todos[0, :, 0]
t = np.arange(eeg.shape[2]) / FS

# espectro: janela de 4 s após o início + latência, média dos 8 canais e 15 tentativas
seg = filtrado_todos[:, INICIO + LAT:INICIO + LAT + 4 * FS, :]
NFFT = 4096
amp = np.abs(np.fft.rfft(seg, n=NFFT, axis=1)).mean(axis=(0, 2))
fr = np.fft.rfftfreq(NFFT, 1 / FS)
sel = (fr >= 3) & (fr <= 40)

AZUL, TXT, TXT2, GRADE = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": TXT2, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.titlecolor": TXT})
fig, ax = plt.subplots(3, 1, figsize=(9, 8.2), facecolor="#fcfcfb")
for x in ax:
    x.set_facecolor("#fcfcfb"); x.grid(color=GRADE, lw=0.8); x.set_axisbelow(True)
    for lado in ("top", "right"):
        x.spines[lado].set_visible(False)

ax[0].plot(t, bruto, color=AZUL, lw=1.2)
ax[0].set_title("1. Sinal bruto do arquivo (canal 1, tentativa 1): valores grandes por causa do offset do aparelho", loc="left", fontsize=10)
ax[0].set_ylabel("valor bruto")

ax[1].plot(t, filtrado, color=AZUL, lw=1.2)
ax[1].axvline(INICIO / FS, color=TXT2, ls="--", lw=1)
ax[1].text(INICIO / FS + 0.03, ax[1].get_ylim()[1] * 0.85, "tecla começa a piscar", color=TXT2, fontsize=9)
ax[1].set_title("2. Depois do filtro 6–80 Hz: offset e ruído lento removidos, sinal oscila em torno de zero", loc="left", fontsize=10)
ax[1].set_xlabel("tempo (s)"); ax[1].set_ylabel("amplitude")

ax[2].plot(fr[sel], amp[sel], color=AZUL, lw=2)
for h, nome in ((1, f"{f0} Hz (tecla '{letra}')"), (2, f"{2 * f0} Hz (2º harmônico)")):
    ax[2].axvline(h * f0, color=TXT2, ls="--", lw=1)
    ax[2].text(h * f0 + 0.4, amp[sel].max() * 0.9, nome, color=TXT, fontsize=9)
ax[2].set_title("3. Espectro (FFT, média de 8 canais × 15 tentativas): pico na frequência da tecla olhada", loc="left", fontsize=10)
ax[2].set_xlabel("frequência (Hz)"); ax[2].set_ylabel("amplitude")

fig.suptitle(f"Sujeito {s}, olhando para a tecla '{letra}' ({f0} Hz)", x=0.01, ha="left", fontsize=12, color=TXT)
fig.tight_layout()
nome_arq = f"sinal_s{s}_{'espaco' if letra == ' ' else letra}.png"
fig.savefig(nome_arq, dpi=150)
print(f"figura salva em {nome_arq}")
print(f"pico do espectro: {fr[sel][np.argmax(amp[sel])]:.2f} Hz (esperado: {f0} Hz)")
