import h5py
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    in_fs = ["COCO/features/resplit_val_obj36.h5","COCO/features/train2014_obj36.h5", "COCO/features/val2014_obj36.h5", "VG/features/vg_gqa_obj36.h5"]
    out_fs = ["COCO/features/converted_resplit_val_obj36.h5","COCO/features/converted_train2014_obj36.h5", "COCO/features/converted_val2014_obj36.h5", "VG/features/converted_vg_gqa_obj36.h5"]

    for in_f, out_f in zip(in_fs, out_fs):
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

