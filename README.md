# Soletrador SSVEP com rede neural

A rede lê o EEG de uma pessoa olhando para teclas que piscam em frequências diferentes e escreve o texto.
A base usada é a de Nakanishi et al. (2015): 10 sujeitos, 12 teclas (9,25 a 14,75 Hz), 8 canais, 256 Hz.

## Instalação

```bash
pip install numpy scipy torch
```

## Uso (3 passos)

```bash
python baixar_dados.py          # baixa dados/s1.mat ... s10.mat (~150 MB)
python treinar.py --janela 2.0  # treina com os sujeitos 1–8, testa nos sujeitos 9 e 10, salva modelo.pt
python soletrar.py "ola mundo"  # a rede escreve a frase a partir do EEG do sujeito 9
```

Opções úteis:

- `python soletrar.py "ola mundo" --sujeito 10`: usa o outro sujeito de teste
- `python soletrar.py "ola mundo" --metodo fbcca`: compara com o método clássico sem treino
- `python soletrar.py "ola mundo" --rapido`: não espera o tempo real entre as letras
- `python treinar.py --janela 1.0`: janela menor, mais rápido e com mais erros
- `python treinar.py --loso`: validação completa (treina em 9, testa no restante, 10 rodadas; ~15 min)

Letras disponíveis (a base só tem 12 teclas): espaço `a d e i l m n o r s u`.

## Divisão treino/teste

A base não vem dividida. A divisão é por pessoa, para testar em alguém que a rede nunca viu:

- treino: sujeitos 1–8
- teste: sujeitos 9 e 10

## Resultados (CPU)

| Avaliação | Janela | FBCCA | C-CNN |
|---|---|---|---|
| Teste S9 e S10 | 1 s | 63,9% | 78,6% |
| Teste S9 e S10 | 2 s | 93,9% | 96,1% |
| LOSO (10 rodadas) | 0,5 s | 24,2% | 39,4% |
| LOSO (10 rodadas) | 1 s | 57,4% | 73,6% |
| LOSO (10 rodadas) | 2 s | 84,6% | 92,4% |

Demo com 2 s: "ola mundo sou um ser da rede neural" saiu sem erros para o sujeito 9 e com 2 erros para o sujeito 10.

## Arquivos

- `ssvep.py`: leitura da base, filtros, FBCCA e a rede C-CNN
- `baixar_dados.py`: download da base
- `treinar.py`: treino e avaliação
- `soletrar.py`: demonstração do soletrador
- `mostrar_shape.py`: abre um arquivo da base e mostra o shape (`python mostrar_shape.py 1`)
- `mostrar_sinal.py`: figura com o sinal bruto, o filtrado e o espectro (`python mostrar_sinal.py`; precisa de matplotlib)
- `explorar_dados.ipynb`: notebook que carrega a base passo a passo e mostra os shapes (abrir com `jupyter notebook`)
