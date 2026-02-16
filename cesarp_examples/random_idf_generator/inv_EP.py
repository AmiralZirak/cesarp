
import os
import pandas as pd
import numpy as np
import cesarp.common
import multiprocessing as mp


def read_and_reshape_csv(file_path):
    max_columns = 9
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path,header=None)

    # Determine the current number of columns
    current_columns = df.shape[1]

    # If the DataFrame has fewer columns than max_columns, pad with NaN
    if current_columns < max_columns:
        # Pad the DataFrame with NaN values to ensure it has max_columns
        padding = pd.DataFrame(-999, index=np.arange(len(df)), columns=np.arange(current_columns, max_columns))
        df = pd.concat([df, padding], axis=1)

    # Reshape the DataFrame to a 3D tensor with shape (24, 365, max_columns)
    reshaped_array = df.to_numpy().reshape(365, 24, max_columns)

    return reshaped_array


def concatenate_tensors(tensors):
    # Concatenate all tensors along a new fourth dimension
    return np.stack(tensors, axis=-1)

def __abs_path(path):
    return cesarp.common.abs_path(path, os.path.abspath(__file__))

import multiprocessing as mp

# Function to read, reshape, and return the tensor
def process_file(path):
    reshaped_tensor = read_and_reshape_csv(path)
    return reshaped_tensor

if __name__ == '__main__':
    # List to store data frames
    file_paths = []
    file_selection = [625,999,2677,3416,5365,5794,6619,6862,7805,
                      7897,8380,8779,9142,9505,9873,10165,10568,
                      10850,11079,11845,12964,15339,15516,16686,
                      16790,17188,17506,17599,18285,18990,20174,
                      20858,22632,24808,24822,25130,25998,26641,
                      27699,30743,32150,33243,34104,34445,35061,
                      35209,35239,36070,36102,37303,37353,37640,
                      38516,40524,41130,42120,42757,42835,44560,
                      45887,46224,46986,49343,49438,50456,50740,
                      52063,52509,53663,53732,54001,54367,54714,
                      55029,55065,55135,55385,55829,55940,56209,
                      57499,57824,58356,59129,59154,59403,59939,
                      63945,64781,65430,66466,68333,69833,72010,
                      73591,76763,77204,77382,79353,79880,80454,
                      80621,83741,83835,84146,84309,84547,85969,
                      86637,88106,89871,90053,90241,93178,93240,
                      93399,93863,94239,95176,95491,95817,98157,
                      99202]
    # 1st try 125 iteration : file_selection = [865,979,1343,2180,3341,6012,6075,6498,6613,7516,7636,7646,7882,8804,
                      #9103,9777,9850,10116,11974,13768,15285,15527,16046,16283,16558,17025,
                      #17297,19209,21401,21697,21719,21955,24952,25214,25284,25627,27841,
                      #30107,31323,31984,33863,35153,35342,35667,38101,38216,38231,38761,
                      #39103,39167,39954,40485,41261,42024,42863,43065,44052,45170,46190,
                      #46807,47883,47993,49399,49511,49615,50022,50962,51279,51759,51802,
                      #53334,55231,56707,57322,57480,58012,60513,60785,61147,61704,62043,
                      #63916,64317,64982,66458,66900,68494,68907,70627,72141,75489,76645,
                      #76698,76790,77073,77462,78855,79324,80538,81201,83094,84575,84630,
                      #84832,84955,85579,85709,86136,86493,87132,87334,88435,89105,89621,
                      #91548,93575,96807,97089,97662,98060,98498,98532,98642,98887,99502]

    for i in file_selection:  # 99856 (1,10001) (10001,20001)
        file_name = (f"/home/amirali/Desktop/Amirali/CESARP/simdata/fid_{i}.csv")
        print(file_name,"***************")
        file_paths.append(file_name)

    # List to store the resulting tensors
    tensors = []

    # Use multiprocessing.Pool to parallelize the process
    with mp.Pool() as pool:
        # Map the file paths to the process_file function in parallel
        results = pool.map(process_file, file_paths)

        # Extend the tensors list with the results
        tensors.extend(results)

    # List of file paths for CSVs
    # file_paths = ['path_to_file1.csv', 'path_to_file2.csv', 'path_to_filem.csv']

    # Determine the maximum number of columns in any file
    # max_columns = 9  # max(pd.read_csv(f, nrows=0).shape[1] for f in file_paths)

    # List to hold each reshaped tensor
    # tensors = []

    # for path in file_paths:
    #     reshaped_tensor = read_and_reshape_csv(path)
    #     tensors.append(reshaped_tensor)

    # Concatenate all reshaped tensors
    final_tensor = concatenate_tensors(tensors)

    # Save the final tensor to a binary file using numpy
    # save_path = __abs_path(f"./results/example/final_tensor.npy")
    save_path = __abs_path(f"./second_samples.npy")
    np.save(save_path, final_tensor)

    print("Tensor saved successfully with shape:", final_tensor.shape)
