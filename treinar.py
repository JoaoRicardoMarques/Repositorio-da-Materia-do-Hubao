"""Treina a C-CNN e avalia em sujeitos separados para teste.

Divisão padrão (inter-sujeito):
  treino = sujeitos 1–8  |  teste = sujeitos 9 e 10 (nunca vistos pela rede)
Com --loso roda a validação completa "deixa um sujeito de fora" (10 rodadas).

Uso:
  python treinar.py                 # janela de 1 s, salva modelo.pt
  python treinar.py --janela 2.0
  python treinar.py --loso
"""
import argparse
import time
import numpy as np
import torch
import ssvep as S

ap = argparse.ArgumentParser()
ap.add_argument("--janela", type=float, default=1.0, help="segundos de EEG por decisão")
ap.add_argument("--epocas", type=int, default=12)
ap.add_argument("--treino", default="1-8", help="sujeitos de treino, ex.: 1-8")
ap.add_argument("--teste", default="9-10", help="sujeitos de teste, ex.: 9-10")
ap.add_argument("--loso", action="store_true", help="validação leave-one-subject-out")
ap.add_argument("--dados", default="dados")
a = ap.parse_args()


def faixa(txt):
    i, j = map(int, txt.split("-")) if "-" in txt else (int(txt), int(txt))
    return list(range(i, j + 1))


n = int(a.janela * S.FS)
D = {s: S.carregar(s, a.dados) for s in range(1, 11)}
T_sel = a.janela + 0.5  # tempo por caractere na ITR (janela + 0,5 s para mover o olhar)


def avaliar(treino, teste, verbose=True):
    X = np.concatenate([D[s][0] for s in treino])
    y = np.concatenate([D[s][1] for s in treino])
    t0 = time.time()
    m = S.treinar_ccnn(X, y, n, epocas=a.epocas, log=print if verbose else None)
    if verbose:
        print(f"  treino levou {time.time() - t0:.0f} s")
    linhas = []
    for s in teste:
        Xt, yt = D[s]
        acc_cnn = np.mean(S.prever_ccnn(m, Xt, n) == yt)
        acc_cca = np.mean([S.fbcca(x, n) == t for x, t in zip(Xt, yt)])
        linhas.append((s, acc_cca, acc_cnn))
    return m, linhas


def imprimir(linhas):
    print(f"\n{'sujeito':>8} {'FBCCA':>8} {'C-CNN':>8}")
    for s, c, r in linhas:
        print(f"{s:>8} {c * 100:7.1f}% {r * 100:7.1f}%")
    c = np.mean([l[1] for l in linhas]); r = np.mean([l[2] for l in linhas])
    print(f"{'média':>8} {c * 100:7.1f}% {r * 100:7.1f}%")
    print(f"{'ITR':>8} {S.itr(c, T_sel):7.1f}  {S.itr(r, T_sel):7.1f}   bits/min  (janela {a.janela} s)")


if a.loso:
    todas = []
    for s in range(1, 11):
        print(f"rodada: teste = sujeito {s}", flush=True)
        _, l = avaliar([k for k in range(1, 11) if k != s], [s], verbose=False)
        todas += l
    imprimir(todas)
else:
    treino, teste = faixa(a.treino), faixa(a.teste)
    print(f"treino: sujeitos {treino} | teste: sujeitos {teste} | janela {a.janela} s")
    m, l = avaliar(treino, teste)
    imprimir(l)
    torch.save({"estado": m.state_dict(), "janela": a.janela, "teste": teste}, "modelo.pt")
    print("\nmodelo salvo em modelo.pt")
