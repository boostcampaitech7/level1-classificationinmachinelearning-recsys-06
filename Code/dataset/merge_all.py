from typing import List, Dict
import os
import pandas as pd
from tqdm import tqdm

def merge_all(data_path:str, export_filename:str ="raw.csv"):    
    # Train Test 불러오기
    train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train") # train 에는 _type = train 
    test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test") # test 에는 _type = test
    #Train Test 합치기
    df: pd.DataFrame = pd.concat([train_df, test_df], axis=0)
    del train_df,test_df
    
    file_names: List[str] = [
        f for f in os.listdir(data_path) if f.startswith("HOURLY_") and f.endswith(".csv")
    ]

    # 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
    file_dict: Dict[str, pd.DataFrame] = {
        f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names
    }

    for _file_name, _df in tqdm(file_dict.items()):
        # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
        _rename_rule = {
            col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
            for col in _df.columns
        }
        _df = _df.rename(_rename_rule, axis=1)
        df = df.merge(_df, on="ID", how="left")
    df.to_csv(os.path.join(data_path, export_filename),index=False)
    del file_names,file_dict
    return df
    