import pandas as pd
import os
from glob import glob

def load_data(folder_name : str):

    folder = ["1.회원정보", "2.신용정보", "3.승인매출정보", "4.청구입금정보", "5.잔액정보", "6.채널정보", "7.마케팅정보", "8.성과정보"]
    assert folder_name in folder, f"folder_name must be one of {folder}"

    folder_train_path = f'../../dataset/train/{folder_name}/'
    folder_test_path = f'../../dataset/test/{folder_name}/'

    parquet_train_files = glob(os.path.join(folder_train_path, '*.parquet'))
    parquet_test_files = glob(os.path.join(folder_test_path, '*.parquet'))

    df = pd.DataFrame()
    test_df = pd.DataFrame()

    for file in parquet_train_files:
        _df = pd.read_parquet(file, engine='fastparquet')
        df = pd.concat([df, _df], ignore_index=True)
        print(f"✅ File: {file} Completed!")

    print(f"🔹 Shape : {df.shape}")
    print()

    for file in parquet_test_files:
        _df = pd.read_parquet(file, engine='fastparquet')
        test_df = pd.concat([test_df, _df], ignore_index=True)
        print(f"✅ File: {file} Completed!")

    print(f"🔹 Shape : {test_df.shape}")

    return df, test_df