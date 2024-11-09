import pandas as pd
from dask import dataframe
import sys
from os import listdir, path, mkdir


def process_zinc(pf_zinc, df_out):
    #check empty file
    if path.getsize(pf_zinc) == 0:
        return df_out
    
    df = pd.read_csv(pf_zinc, sep='\t', low_memory=False)

    # here recreate out in case of empty
    if df_out.shape == (0,0):
        return df
    else:
        df_out = pd.concat([df_out, df], axis=0, ignore_index=True)
    return df_out



## MAIN ##
##########

InDirPath = sys.argv[1]
OutDirName = sys.argv[2]

#create folder -> parquet split in 1 Gb
mkdir(OutDirName)

# create parquet folder and each parquet ~1Gb
l_dir_zinc = listdir(InDirPath)

df_out = pd.DataFrame()
i_dir_zinc = 0
imax_dir_zinc = len(l_dir_zinc)
i_name_parquet = 1

while i_dir_zinc < imax_dir_zinc:
    l_f_zinc = listdir(InDirPath + l_dir_zinc[i_dir_zinc])

    j_file_zinc = 0
    jmax_file_zinc = len(l_f_zinc)
    while j_file_zinc < jmax_file_zinc:
        pf_zinc = "%s%s/%s"%(InDirPath, l_dir_zinc[i_dir_zinc], l_f_zinc[j_file_zinc])

        dtemp = process_zinc(pf_zinc, df_out)
        size1G = int(dtemp.memory_usage().sum()/1e6/1000)

        if size1G > 1:
            df_out.to_parquet("%s/zinc%s.parquet"%(OutDirName, i_name_parquet))
            df_out = pd.read_csv(pf_zinc, sep='\t', low_memory=False)
            i_name_parquet = i_name_parquet + 1
        else:
            df_out = dtemp
        
        j_file_zinc = j_file_zinc + 1
    i_dir_zinc = i_dir_zinc + 1
    