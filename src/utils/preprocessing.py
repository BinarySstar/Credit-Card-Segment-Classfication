import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import time

class NumericTypeOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self, mode='post', verbose=True):
        """
        mode: 'pre' or 'post'
        - 'pre': raw data 대상, float은 보수적으로 유지
        - 'post': 전처리 완료 데이터 대상, float까지 적극적으로 다운캐스팅
        """
        assert mode in ['pre', 'post'], "mode must be 'pre' or 'post'"
        self.mode = mode
        self.verbose = verbose

    def fit(self, X, y=None):
        return self  # 학습할 내용 없음

    def transform(self, X):
        df = X.copy()
        if self.verbose:
            print("Numeric Type Optimizer Transforming...")
        start_mem = df.memory_usage(deep=True).sum() / 1024**2

        for col in df.select_dtypes(include=['number']).columns:
            col_type = df[col].dtypes
            c_min, c_max = df[col].min(), df[col].max()

            if np.issubdtype(col_type, np.integer):
                if c_min >= 0:
                    if c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if np.iinfo(np.int8).min <= c_min <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif np.iinfo(np.int16).min <= c_min <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif np.iinfo(np.int32).min <= c_min <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)

            elif np.issubdtype(col_type, np.floating):
                if self.mode == 'post':
                    if np.finfo(np.float16).min <= c_min <= np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif np.finfo(np.float32).min <= c_min <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                elif self.mode == 'pre':
                    # pre 단계에서는 float을 보수적으로 유지
                    df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        if self.verbose:
            reduction = 100 * (start_mem - end_mem) / start_mem
            print(f"🧠 [mode={self.mode}] 메모리 최적화: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% 감소)")
        return df

class ObjectFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fillna_value='unknown', drop_unique_thresh=1, exclude_columns=None, verbose=True):
        self.fillna_value = fillna_value
        self.drop_unique_thresh = drop_unique_thresh
        self.exclude_columns = exclude_columns if exclude_columns else ['ID','Segment']
        self.cols_to_keep = []
        self.verbose = verbose
    
    def fit(self, X, y=None):
        if self.verbose:
            print("Object Feature Preprocessor Fitting...")
        object_cols = X.select_dtypes(include=['object']).columns
        object_cols = [col for col in object_cols if col not in self.exclude_columns]
        
        for col in object_cols:
            unique_count = X[col].nunique(dropna=False)
            if unique_count > self.drop_unique_thresh:
                self.cols_to_keep.append(col)
        
        if self.verbose:
            print(f"✅ Total object columns: {self.cols_to_keep}")
            print(f"✅ Total object columns to keep: {len(self.cols_to_keep)}")
        return self
    
    def transform(self, X):
        if self.verbose:
            print("Object Feature Preprocessor Transforming...")
        
        # 조기 종료
        if self.cols_to_keep == []:
            if self.verbose:
                print("✅ No object columns to transform.")
            return X
        
        start = time.time()
        x_out = X.copy()

        object_cols = x_out.select_dtypes(include=['object']).columns
        to_drop = [col for col in object_cols if col not in self.cols_to_keep]
        x_out = x_out.drop(columns=to_drop)

        for col in self.cols_to_keep:
            x_out[col] = x_out[col].fillna(self.fillna_value)
            x_out[col] = x_out[col].astype('category')
        
        # encoding
        dummies = pd.get_dummies(x_out[self.cols_to_keep], drop_first=True, dtype=int)
        x_out = x_out.drop(columns=self.cols_to_keep)
        x_out = pd.concat([x_out, dummies], axis=1)
        
        end = time.time()
        if self.verbose:
            print(f"✅ Transformed Complete!")
            print(f"🔹 Transformation Time: {end - start:.2f} seconds")
            print(f"🔹 Shape after transformation: {x_out.shape}")
        return x_out

class NumericFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, drop_unique_thresh=1, exclude_columns=None, verbose=True):
        self.verbose = verbose
        self.exclude_columns = exclude_columns if exclude_columns else ['기준년월']
        self.drop_unique_thresh = drop_unique_thresh
        self.cols_to_keep = []
        self.impute_values = {}

    def fit(self, X, y=None):
        if self.verbose:
            print("Numeric Feature Preprocessor Fitting...")
        num_cols = X.select_dtypes(include=['number']).columns
        num_cols = [col for col in num_cols if col not in self.exclude_columns]

        for col in num_cols:
            unique_count = X[col].nunique(dropna=False)
            if unique_count > self.drop_unique_thresh:
                self.cols_to_keep.append(col)
                if X[col].isnull().any():
                    self.impute_values[col] = X[col].mean()
            
        if self.verbose:
            print(f"✅ Total numeric columns: {self.cols_to_keep}")
            print(f"✅ Total numeric columns to keep: {len(self.cols_to_keep)}")
        return self
    
    def transform(self, X):
        start = time.time()
        if self.verbose:
            print("Numeric Feature Preprocessor Transforming...")
        x_out = X.copy()

        drop_cols = [col for col in x_out.select_dtypes(include=['number']).columns if col not in self.cols_to_keep]
        x_out = x_out.drop(columns=drop_cols)

        for col, mean_value in self.impute_values.items():
            x_out[col] = x_out[col].fillna(mean_value)
        
        end = time.time()
        if self.verbose:
            print(f"✅ Transformed Complete!")
            print(f"🔹 Transformation Time: {end - start:.2f} seconds")
            print(f"🔹 Shape after transformation: {x_out.shape}")
            
        return x_out
        
class DateElapsedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_date='2019-01-01', fillna_value=99999999, exclude_columns=None, verbose=True):
        self.verbose = verbose
        self.fillna_value = fillna_value
        self.base_date = pd.to_datetime(base_date)
        self.exclude_columns = exclude_columns if exclude_columns else ['ID', 'Segment', '기준년월']
        self.date_cols = None

    def fit(self, X, y=None):
        if self.verbose:
            print("Date Elapsed Transformer Fitting...")
        self.date_cols = [col for col in X.columns if any(key in col for key in ['일자', '년월']) and col not in self.exclude_columns]

        if self.verbose:
            print(f"✅ Total date columns: {self.date_cols}")
            print(f"✅ Total date columns to keep: {len(self.date_cols)}")
        return self

    def transform(self, X):
        if self.verbose:
            print("Date Elapsed Transformer Transforming...")
        start = time.time()

        x_out = X.copy()
        for col in self.date_cols:
            col_series = x_out[col]

            parsed_col = pd.to_datetime(col_series, format='%Y%m%d',errors='coerce')
            elapsed_days = (parsed_col - self.base_date).dt.days
            elapsed_col_name = f"{col}_경과일"
            is_na = f"missing_{col}"

            x_out[is_na] = elapsed_days.isna().astype(int)
            x_out[elapsed_col_name] = elapsed_days.fillna(self.fillna_value).astype(int)
        
        x_out = x_out.drop(columns=self.date_cols)
        
        end = time.time()
        if self.verbose:
            print(f"✅ Transformed Complete!")
            print(f"🔹 Transformation Time: {end - start:.2f} seconds")
            print(f"🔹 Shape after transformation: {x_out.shape}")

        return x_out