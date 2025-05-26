# run_all_clients.py
import subprocess
import glob

farm_files = glob.glob("./Data/Farm*.csv")

for path in farm_files:
    print(f"🚀 Lanzando cliente para {path}")
    subprocess.Popen(["python3", "client.py", path])