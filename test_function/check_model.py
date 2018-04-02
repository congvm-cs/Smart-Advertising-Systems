from keras.models import Sequential, load_model
import pandas as pd
import h5py

model_path = '/mnt/c/Users/VMC/Desktop/weights-improvement-20-0.13.hdf5'


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            # print("    Dataset:")
            # for p_name in g.keys():
            #     param = g[p_name]
            #     print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()

# print_structure(model_path)
model = load_model(model_path)

weights = model.get_weights()

print(type(weights))
import numpy as np
weights_arr = np.array(weights)
# import pandas as pd

weight_pd = pd.DataFrame(data=weights)

weight_pd.to_json('/mnt/c/Users/VMC/Desktop/weights_json.json')

print(weights_arr.shape)