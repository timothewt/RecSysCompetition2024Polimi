import pandas as pd
import scipy.sparse as sps

def load_ICM(file_path):
    metadata = pd.read_csv(file_path)

    item_icm_list = metadata['item_id'].tolist()
    feature_list = metadata['feature_id'].tolist()
    weight_list = metadata['data'].tolist()

    return sps.coo_matrix((weight_list, (item_icm_list, feature_list)))