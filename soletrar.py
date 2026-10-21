"""Soletrador SSVEP simulado: a rede "lê" o EEG de um sujeito de teste e escreve o texto.

Para cada letra da frase, pega uma tentativa REAL do sujeito olhando para a tecla
correspondente (que nunca entrou no treino) e passa a janela de EEG para a rede.
O caractere previsto aparece na tela no ritmo de um soletrador de verdade.

Uso:
  python soletrar.py "ola mundo"
  python soletrar.py "ola mundo" --sujeito 9 --metodo fbcca --rapido
Letras disponíveis (12 teclas da base): espaço a d e i l m n o r s u
"""
import argparse
import sys
import time
import numpy as np
import torch
import ssvep as S

ap = argparse.ArgumentParser()
ap.add_argument("frase")
ap.add_argument("--sujeito", type=int, default=None, help="sujeito de teste (padrão: o primeiro do teste)")
ap.add_argument("--metodo", choices=["cnn", "fbcca"], default="cnn")
ap.add_argument("--rapido", action="store_true", help="não esperar o tempo real entre letras")
ap.add_argument("--dados", default="dados")
a = ap.parse_args()

ck = torch.load("modelo.pt")
janela = ck["janela"]
n = int(janela * S.FS)
suj = a.sujeito or ck["teste"][0]
m = S.CCNN(); m.load_state_dict(ck["estado"]); m.eval()

frase = a.frase.lower()
fora = sorted(set(frase) - set(S.TECLAS))
if fora:
    sys.exit(f"letras fora do teclado: {fora}. Disponíveis: {''.join(S.TECLAS)!r}")

X, y = S.carregar(suj, a.dados)
fila = {k: list(np.random.default_rng(0).permutation(np.where(y == k)[0])) for k in range(S.N_ALVOS)}

print(f"sujeito {suj} | método {a.metodo} | janela {janela} s")
print("tecla -> frequência:", ", ".join(f"'{c}'={S.FREQS[i]} Hz" for i, c in enumerate(S.TECLAS)))
print(f"\nalvo : {frase}\nsaída: ", end="", flush=True)

saida, tempos = "", []
for ch in frase:
    k = S.TECLAS.index(ch)
    if not fila[k]:
        fila[k] = list(np.where(y == k)[0])  # reutiliza tentativas se a frase for longa
    trial = X[fila[k].pop(0)]
    if not a.rapido:
        time.sleep(janela + 0.5)  # usuário olha para a tecla durante a janela
    t0 = time.perf_counter()
    pred = S.prever_ccnn(m, trial[None], n)[0] if a.metodo == "cnn" else S.fbcca(trial, n)
    tempos.append(time.perf_counter() - t0)
    saida += S.TECLAS[pred]
    print(S.TECLAS[pred], end="", flush=True)

erros = sum(p != q for p, q in zip(frase, saida))
print(f"\n\nacertos: {len(frase) - erros}/{len(frase)}  |  CER = {100 * erros / len(frase):.1f}%"
      f"  |  decisão em {1000 * np.mean(tempos):.1f} ms/caractere")
