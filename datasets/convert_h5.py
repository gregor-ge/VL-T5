import h5py
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/ukp-storage-1/geigle/misrik/VLT5/datasets/COCO/features/')
    parser.add_argument('--files', type=str, default='val2014-RN50x16-6.h5')

    args = parser.parse_args()
    in_fs = args.files.split(",")
    out_fs = ["converted_"+f for f in in_fs]

    for in_f, out_f in zip(in_fs, out_fs):
        print(in_f, out_f)
        in_f = args.data_dir + in_f
        out_f = args.data_dir + out_f
        f = h5py.File(in_f, 'r', )

        len_data = len(f.keys())

        with h5py.File(out_f, mode="w") as o:
            out_d = []
            for i, k in tqdm(enumerate(f.keys()), total=len_data):
                out_d.append(k)
                for k, v in f[k].items():
                    if k not in o:
                        o.create_dataset(k, data=np.zeros((len_data,)+v[()].shape), dtype=v[()].dtype)
                    o[k][i] = v[()]
            o.create_dataset("img_id", data=out_d)


