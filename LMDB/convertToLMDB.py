import os
import lmdb
import pickle
import argparse
from PIL import Image
import numpy as np
import io

def pack_to_lmdb(cifar_file, lmdb_path):
    if not os.path.exists(cifar_file):
        raise FileNotFoundError(f"CIFAR file not found: {cifar_file}")
    env = lmdb.open(lmdb_path, map_size=int(1e10))
    idx = 0
    with open(cifar_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    images = data[b'data']
    labels = data[b'fine_labels']
    with env.begin(write=True) as txn:
        for img_arr, lbl in zip(images, labels):
            img = Image.fromarray(
                np.reshape(img_arr, (3, 32, 32)).transpose(1, 2, 0)
            )
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            entry = {'image': buf.getvalue(), 'label': lbl}
            txn.put(f"{idx}".encode(), pickle.dumps(entry))
            idx += 1
    env.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='Path to cifar-100-python folder')
    p.add_argument('--train_output', default='lmdb_train')
    p.add_argument('--test_output', default='lmdb_test')
    args = p.parse_args()

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    pack_to_lmdb(os.path.join(args.data_dir, 'train'), args.train_output)
    pack_to_lmdb(os.path.join(args.data_dir, 'test'), args.test_output)