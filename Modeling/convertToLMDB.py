import os, lmdb, pickle
from PIL import Image
import numpy as np
import io
import batch

def pack_to_lmdb(pickle_files, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1e10)
    with env.begin(write=True) as txn:
        idx = 0
        for pf in pickle_files:
            data = batch.load_batch(pf)  # returns dict with 'data' and 'fine_labels'
            for img_arr, lbl in zip(data['data'], data['fine_labels']):
                img = Image.fromarray(img_arr.reshape(3,32,32).transpose(1,2,0))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                entry = {'image': buf.getvalue(), 'label': lbl}
                txn.put(f"{idx}".encode(), pickle.dumps(entry))
                idx += 1
    env.close()

if __name__=='__main__':
    os.makedirs('lmdb_train', exist_ok=True)
    os.makedirs('lmdb_test', exist_ok=True)
    pack_to_lmdb(['train'], 'lmdb_train')
    pack_to_lmdb(['test'], 'lmdb_test')