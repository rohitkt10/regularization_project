import os, sys 
sys.path.append("../..")

from src.download_invivo_data import download, IN_VIVO_NAMES
def main():
    for name in IN_VIVO_NAMES:
        print(f"Downloading data for cell line {name} ...")
        download(
            name=name,
            savepath=f"./{name}",
            shard_size=2000,
        )
        print("*"*60)
        break

if __name__=='__main__':
    main()