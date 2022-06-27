import pandas as pd
import os
from ling_prep import *



os.chdir("C:\\Users\\idanh\\Documents\\תעונ שנה ד\\פרויקט גמר\\New test")


behavioral_df = pd.read_pickle('behavioral_df.pickle')
lingustic_df = pd.read_pickle('full_df.pickle')

lingustic_df.reset_index( drop=True, inplace=True)
lingustic_df = ling_prep(lingustic_df)



