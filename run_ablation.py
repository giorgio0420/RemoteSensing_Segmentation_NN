"""
Esperimento centrale della consegna:
  "compare model performance with and without pretraining to quantify its benefit"

Lancia una griglia (modalita' di pretraining) x (quantita' di dati = scenario data-scarce)
e raccoglie i risultati in results_summary.csv. Ogni run gira in un PROCESSO separato
cosi' la memoria GPU viene liberata tra un run e l'altro.

Uso:
  python run_ablation.py                 # griglia di default (veloce)
  python run_ablation.py --full          # aggiunge anche il run su tutto il dataset
"""
import argparse
import subprocess
import sys

# (modalita', etichetta). "scratch" = senza pretraining; "rsp" = pretraining satellitare.
MODES = ["scratch", "rsp"]            # aggiungi "imagenet" e/o "satmaepp" se vuoi
SUBSETS = [100, 300]                  # asse "data-scarce" (numero immagini di train)
EPOCHS = 20


VAL_CAP = 400  # cap sul val SOLO durante la griglia, per velocita'


def run(mode, subset, epochs):
    tag = f"{mode}_n{subset}"
    cmd = [sys.executable, "main.py", "--mode", mode, "--train-subset", str(subset),
           "--val-subset", str(VAL_CAP), "--epochs", str(epochs), "--tag", tag]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="aggiunge anche il run su tutto il train")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    for subset in SUBSETS:
        for mode in MODES:
            run(mode, subset, args.epochs)

    if args.full:
        for mode in MODES:
            cmd = [sys.executable, "main.py", "--mode", mode, "--epochs", str(args.epochs),
                   "--tag", f"{mode}_full"]
            print(f"\n>>> {' '.join(cmd)}")
            subprocess.run(cmd, check=False)

    print("\n========== ABLATION COMPLETATA ==========")
    print("Apri results_summary.csv per la tabella finale (best mIoU per ogni run).")
