try:
    import urllib.request as request
except ImportError:
    raise ImportError('You should use Python 3.x')
import gzip, os, os.path

save_path = "./sample/"
base_url = "http://yann.lecun.com/exdb/mnist"
file_names = [
    "train-images-idx3-ubyte.gz",  # training data
    "train-labels-idx1-ubyte.gz",  # training label
    "t10k-images-idx3-ubyte.gz",  # test data
    "t10k-labels-idx1-ubyte.gz"  # test label
]

# download
if not os.path.exists(save_path): os.mkdir(save_path)
for file_name in file_names:
    url = base_url + "/" + file_name
    location = save_path + "/" + file_name
    print("download : " + url)
    if os.path.exists(location):
        print("files exist! skipped to download file")
    else:
        request.urlretrieve(url, location)
        print("downloading is completed!")

# unzip
for file_name in file_names:
    gz_file = save_path + "/" + file_name
    raw_file = save_path + "/" + file_name.replace(".gz", "")
    print("gzip : " + file_name)
    with gzip.open(gz_file, "rb") as fp:
        body = fp.read()
        with open(raw_file, "wb") as w:
            w.write(body)

print("unzip is completed!")
