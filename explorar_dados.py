"""Explora a base de Nakanishi et al. (2015) passo a passo e mostra os shapes.
Mesmo conteúdo do explorar_dados.ipynb, em script Python.

Uso:
  python explorar_dados.py
Precisa de numpy, scipy e matplotlib, e da pasta dados/ (rode antes o baixar_dados.py).
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import cheby1, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def titulo(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# ---------------------------------------------------------------- 1. carregamento
titulo("1. Carregamento do participante 1 (dados/s1.mat)")
raw_data = loadmat("dados/s1.mat")
print("chaves:", list(raw_data.keys()))

titulo("2. Shape de cada variável do arquivo")
for k, v in raw_data.items():
    if not k.startswith("__"):
        print(f"  {k:10s} {v.shape}")
print("\nA variável principal é 'eeg'. As outras são sobras do MATLAB")
print("('train' tem os mesmos 8 canais de 'eeg' + 10 extras não usados).")

# ---------------------------------------------------------------- 2. info básica
titulo("3. Informações básicas")
fs = 256                                    # não vem no arquivo: README/artigo
freqs = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
teclas = list(" adeilmnorsu")

eeg = raw_data["eeg"]
n_alvos, n_canais, n_amostras, n_tentativas = eeg.shape
print(f"Taxa de amostragem: {fs} Hz")
print(f"Número de canais: {n_canais} (região occipital)")
print(f"Número de alvos: {n_alvos}")
print(f"Amostras por tentativa: {n_amostras} ({n_amostras / fs:.2f} s)")
print(f"Tentativas por alvo: {n_tentativas}")
print("\nAlvo -> caractere -> frequência:")
for i, (c, f) in enumerate(zip(teclas, freqs)):
    print(f"  {i:2d} -> {repr(c)} -> {f} Hz")

titulo("4. Shape da matriz principal e exemplo de dados brutos")
print("eeg:", eeg.shape, "-> (alvos, canais, amostras, tentativas)")
print("\nalvo 8 ('o', 14,25 Hz), canal 1, tentativa 1, primeiras 10 amostras:")
print(eeg[8, 0, :10, 0])
print("(valores grandes = offset do aparelho; o filtro remove)")

# ---------------------------------------------------------------- 3. organização
titulo("5. Organização em (n_tentativas, n_amostras, n_canais)")
# o arquivo já vem com cada tentativa recortada e do mesmo tamanho: basta reordenar os eixos
data_X = eeg.transpose(3, 0, 2, 1).reshape(-1, n_amostras, n_canais)
data_y = np.tile(np.arange(n_alvos), n_tentativas)
print("data_X:", data_X.shape)
print("data_y:", data_y.shape)
print("primeiros rótulos:", data_y[:24])
print("\nconferência (a reorganização não embaralhou nada):",
      np.allclose(data_X[0], eeg[0, :, :, 0].T), np.allclose(data_X[13], eeg[1, :, :, 1].T))

titulo("6. Recorte do período de estímulo")
inicio = 39 + int(round(0.135 * fs))   # início do estímulo (0,15 s) + latência visual = amostra 74
janela = 4 * fs                         # 4 s = 1024 amostras
data_X_estim = data_X[:, inicio:inicio + janela, :]
print("início do recorte (amostra):", inicio)
print("data_X após o recorte:", data_X_estim.shape)

# ---------------------------------------------------------------- 4. todos os participantes
titulo("7. Todos os participantes")
X_todos, y_todos, sujeito = [], [], []
for s in range(1, 11):
    e = loadmat(f"dados/s{s}.mat")["eeg"]
    X_s = e.transpose(3, 0, 2, 1).reshape(-1, e.shape[2], e.shape[1])[:, inicio:inicio + janela, :]
    X_todos.append(X_s)
    y_todos.append(np.tile(np.arange(e.shape[0]), e.shape[3]))
    sujeito.append(np.full(len(X_s), s))
X_todos = np.concatenate(X_todos)
y_todos = np.concatenate(y_todos)
sujeito = np.concatenate(sujeito)
print("X_todos:", X_todos.shape)
print("y_todos:", y_todos.shape)
print("sujeito:", sujeito.shape)
print("tentativas por alvo:", np.bincount(y_todos))

titulo("8. Divisão treino / teste (por participante)")
treino, teste = sujeito <= 8, sujeito >= 9
print("X_treino:", X_todos[treino].shape, "| y_treino:", y_todos[treino].shape, " (participantes 1-8)")
print("X_teste: ", X_todos[teste].shape, "| y_teste: ", y_todos[teste].shape, " (participantes 9-10)")

# ---------------------------------------------------------------- 5. visualização
titulo("9. Visualização: bruto, filtrado e espectro (participante 9, alvo 'o')")
alvo = 8
e9 = loadmat("dados/s9.mat")["eeg"]
b, a = cheby1(4, 0.5, [6 / (fs / 2), 80 / (fs / 2)], btype="band")
filtrado = filtfilt(b, a, e9[alvo], axis=1)
seg = filtrado[:, inicio:inicio + janela, :]
amp = np.abs(np.fft.rfft(seg, n=4096, axis=1)).mean(axis=(0, 2))
fr = np.fft.rfftfreq(4096, 1 / fs)
sel = (fr >= 3) & (fr <= 40)

t = np.arange(e9.shape[2]) / fs
fig, ax = plt.subplots(3, 1, figsize=(9, 7.5))
ax[0].plot(t, e9[alvo, 0, :, 0], color="#2a78d6", lw=1); ax[0].set_title("Bruto (canal 1, tentativa 1)", loc="left")
ax[1].plot(t, filtrado[0, :, 0], color="#2a78d6", lw=1); ax[1].set_title("Filtrado 6–80 Hz", loc="left"); ax[1].set_xlabel("tempo (s)")
ax[2].plot(fr[sel], amp[sel], color="#2a78d6", lw=2); ax[2].set_title("Espectro (média de 8 canais x 15 tentativas)", loc="left")
for h in (1, 2):
    ax[2].axvline(h * freqs[alvo], color="gray", ls="--", lw=1)
ax[2].set_xlabel("frequência (Hz)")
for x in ax:
    x.spines[["top", "right"]].set_visible(False); x.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("explorar_dados.png", dpi=150)
print(f"pico do espectro: {fr[sel][np.argmax(amp[sel])]:.2f} Hz (esperado: {freqs[alvo]} Hz)")
print("figura salva em explorar_dados.png")

# ---------------------------------------------------------------- 6. salvamento (opcional)
# np.save('X_todos.npy', X_todos)
# np.save('y_todos.npy', y_todos)
# np.save('sujeito.npy', sujeito)
