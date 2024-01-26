# data_manager.py

import pandas as pd
import io
import xlsxwriter

class DataManager:
    @staticmethod
    def convert_df_to_xlsx(df):
        """
        Converts a pandas DataFrame to an XLSX file.
        :param df: DataFrame to convert.
        :return: XLSX file in bytes.
        """
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
        return output.getvalue()

    @staticmethod
    def merge_dataframes(dfs, merge_strategy='outer'):
            for df in dfs:
                if not isinstance(df, (pd.DataFrame, pd.Series)):
                    raise TypeError("All elements must be DataFrame or Series objects")
            return pd.concat(dfs, axis=0, join=merge_strategy, ignore_index=True)

    @staticmethod
    def preprocess_dataframe(df, rename_mappings=None, convert_columns=None):
        """
        Performs preprocessing steps on a DataFrame, like renaming columns and converting values.
        :param df: DataFrame to preprocess.
        :param rename_mappings: Dictionary of column rename mappings.
        :param convert_columns: Dictionary of columns with their respective conversion functions.
        :return: Preprocessed DataFrame.
        """
        if rename_mappings:
            df.rename(columns=rename_mappings, inplace=True)

        if convert_columns:
            for col, func in convert_columns.items():
                df[col] = df[col].apply(func)

        return df
