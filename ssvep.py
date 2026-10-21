"""Funções comuns: leitura da base, pré-processamento, FBCCA e a rede C-CNN.

Base: Nakanishi et al. (2015) — 12 alvos JFPM, 10 sujeitos, 8 canais, 256 Hz,
15 tentativas por alvo. https://github.com/mnakanishi/12JFPM_SSVEP
"""
import numpy as np
import scipy.io as sio
from scipy.signal import cheby1, filtfilt
import torch
import torch.nn as nn

FS = 256
FREQS = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                  10.25, 12.25, 14.25, 10.75, 12.75, 14.75])  # Hz, na ordem dos alvos
N_ALVOS, N_CANAIS = 12, 8
ONSET = 39 + int(round(0.135 * FS))  # início do estímulo (0,15 s) + latência visual (~135 ms)

# 12 alvos -> 12 "teclas". Alfabeto reduzido (a base tem só 12 alvos).
TECLAS = list(" adeilmnorsu")


# ------------------------------------------------------------------ dados
def carregar(sujeito, pasta="dados"):
    """Retorna X (180 tentativas, 8 canais, 1114 amostras) e y (alvo 0..11)."""
    eeg = sio.loadmat(f"{pasta}/s{sujeito}.mat")["eeg"]  # (12, 8, 1114, 15)
    X = eeg.transpose(3, 0, 1, 2).reshape(-1, N_CANAIS, eeg.shape[2])
    y = np.tile(np.arange(N_ALVOS), eeg.shape[3])
    return X.astype(np.float32), y


def passa_faixa(X, lo=6, hi=80):
    b, a = cheby1(4, 0.5, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, X, axis=-1).astype(np.float32)


def recortes(X, y, n, passo):
    """Aumento de dados: várias janelas de n amostras por tentativa."""
    inicios = range(ONSET, X.shape[-1] - n + 1, passo)
    return (np.concatenate([X[..., s:s + n] for s in inicios]),
            np.concatenate([y for _ in inicios]))


# ------------------------------------------------------------------ FBCCA (sem treino)
_FB = [cheby1(4, 0.5, [8 * (m + 1) / (FS / 2), 88 / (FS / 2)], btype="band") for m in range(5)]
_W = np.array([(m + 1) ** -1.25 + 0.25 for m in range(5)])


def _refs(f, n, H=5):
    t = np.arange(n) / FS
    return np.vstack([fn(2 * np.pi * h * f * t) for h in range(1, H + 1) for fn in (np.sin, np.cos)])


def _cca(X, Y):
    Qx, _ = np.linalg.qr((X - X.mean(1, keepdims=True)).T)
    Qy, _ = np.linalg.qr((Y - Y.mean(1, keepdims=True)).T)
    return np.linalg.svd(Qx.T @ Qy, compute_uv=False)[0]


def fbcca(x, n):
    """x: (8 canais, amostras) de uma tentativa inteira. Retorna o alvo previsto."""
    subs = [filtfilt(b, a, x, axis=1)[:, ONSET:ONSET + n] for b, a in _FB]
    rho = np.array([[_cca(s, _refs(f, n)) ** 2 for f in FREQS] for s in subs])
    return int(np.argmax(_W @ rho))


# ------------------------------------------------------------------ C-CNN
NFFT = 1024                      # resolução espectral de 0,25 Hz
LO, HI = int(3 * NFFT / FS), int(35 * NFFT / FS)   # faixa 3–35 Hz


def espectro(seg, bloco=2000):
    """Partes real e imaginária da FFT: (N, 8, n) -> (N, 8, 2*(HI-LO))."""
    out = []
    for i in range(0, len(seg), bloco):
        F = np.fft.rfft(seg[i:i + bloco], n=NFFT, axis=-1)[..., LO:HI] / seg.shape[-1]
        out.append(np.concatenate([F.real, F.imag], axis=-1).astype(np.float32))
    return np.concatenate(out)


class CCNN(nn.Module):
    """CNN no domínio da frequência (estilo C-CNN, Ravi et al. 2020)."""
    def __init__(self, C=N_CANAIS, L=2 * (HI - LO), K=N_ALVOS, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 2 * C, (C, 1)), nn.BatchNorm2d(2 * C), nn.ReLU(), nn.Dropout(drop),   # espacial
            nn.Conv2d(2 * C, 2 * C, (1, 10)), nn.BatchNorm2d(2 * C), nn.ReLU(), nn.Dropout(drop),  # espectral
            nn.Flatten(), nn.Linear(2 * C * (L - 9), K))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def treinar_ccnn(X, y, n, epocas=12, passo=0.25, semente=0, log=None):
    torch.manual_seed(semente)
    Xc, yc = recortes(passa_faixa(X), y, n, int(passo * FS))
    Xf, yf = torch.tensor(espectro(Xc)), torch.tensor(yc)
    m = CCNN()
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-3)
    perda = nn.CrossEntropyLoss()
    for ep in range(epocas):
        m.train()
        perm = torch.randperm(len(yf))
        tot = 0.0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            L = perda(m(Xf[idx]), yf[idx])
            L.backward()
            opt.step()
            tot += L.item() * len(idx)
        if log:
            log(f"  época {ep + 1:2d}/{epocas}  perda {tot / len(yf):.3f}")
    return m.eval()


def prever_ccnn(m, X, n):
    with torch.no_grad():
        return m(torch.tensor(espectro(passa_faixa(X)[..., ONSET:ONSET + n]))).argmax(1).numpy()


def itr(P, T, N=N_ALVOS):
    """Taxa de transferência de informação (bits/min). T = janela + troca do olhar."""
    if P <= 1 / N:
        return 0.0
    P = min(P, 1 - 1e-9)
    return 60 / T * (np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))
