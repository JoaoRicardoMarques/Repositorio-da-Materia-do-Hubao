"""Baixa a base de Nakanishi et al. (2015) para a pasta dados/ (~150 MB)."""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/mnakanishi/12JFPM_SSVEP/master/data/s{}.mat"
os.makedirs("dados", exist_ok=True)
for s in range(1, 11):
    destino = f"dados/s{s}.mat"
    if os.path.exists(destino) and os.path.getsize(destino) > 1_000_000:
        print(f"s{s}.mat já existe")
        continue
    print(f"baixando s{s}.mat ...", flush=True)
    urllib.request.urlretrieve(URL.format(s), destino)
print("pronto: dados/s1.mat ... dados/s10.mat")
