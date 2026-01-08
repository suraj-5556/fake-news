import pandas as pd
def combine_text_columns(x):
        return x.astype(str).agg(" ".join, axis=1)  