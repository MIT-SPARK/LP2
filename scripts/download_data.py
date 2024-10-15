import gdown
import os

if not os.path.exists("data"):
    os.mkdir("data")

_ = gdown.download_folder(
    "https://drive.google.com/drive/folders/1-ThhNFVzWQvtsCLv1fo0R7Oo8fTX5uMT",
    output="data/",
    quiet=True,
)

share_dir = "data/share"
if not os.path.exists(share_dir):
    os.mkdir(share_dir)
share_dir_abs = os.path.abspath(share_dir)

os.system(f"tar -xf {share_dir_abs}/home.tar.gz -C data/")
os.system(f"tar -xf {share_dir_abs}/office.tar.gz -C data/")