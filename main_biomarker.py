""" Example for biomarker identification
"""
import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    data_folder = 'BRCA'
    model_folder = os.path.join(data_folder, 'models')
    view_list = [1,2,3]
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5

    featimp_list_list = []
    for rep in range(5):
        featimp_list = cal_feat_imp(data_folder, os.path.join(model_folder, str(rep+1)), 
                                    view_list, num_class)
        featimp_list_list.append(copy.deepcopy(featimp_list))


    summarize_imp_feat(featimp_list_list)
    

#%%
import numpy as np
import pandas as pd

def summarize_imp_feat(featimp_list_list, topn=30):
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])
    df_tmp_list = []
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
        df_tmp_list.append(df_tmp.copy(deep=True))
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    
    for r in range(1,num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
            #df_featimp = df_featimp.append(df_tmp.copy(deep=True), ignore_index=True)
            df_featimp = pd.concat([df_featimp, df_tmp.copy(deep=True)], ignore_index=True)
    
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()
    df_featimp_top = df_featimp_top.sort_values(by='imp',ascending=False)
    df_featimp_top = df_featimp_top.iloc[:topn]
    print('{:}\t{:}'.format('Rank','Feature name'))
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i+1,df_featimp_top.iloc[i]['feat_name']))
    return df_featimp_top

df_featimp_top=summarize_imp_feat(featimp_list_list)
df_featimp_top.to_csv(f'{data_folder}_feat_importance.csv', index=False)

# Rank	Feature name
# 1	SOX11|6664
# 2	hsa-mir-205
# 3	GPR37L1
# 4	AMY1A|276
# 5	SLC6A15|55117
# 6	FABP7|2173
# 7	MIR563
# 8	SLC6A14|11254
# 9	hsa-mir-187
# 10	SLC6A2|6530
# 11	FGFBP1|9982
# 12	DSG1|1828
# 13	UGT8|7368
# 14	ANKRD45|339416
# 15	OR1J4
# 16	ATP10B
# 17	PI3|5266
# 18	hsa-mir-452
# 19	hsa-mir-20b
# 20	SERPINB5|5268
# 21	KRTAP3-3
# 22	COL11A2|1302
# 23	hsa-mir-224
# 24	FLJ41941
# ...
# 27	TMEM207
# 28	CDH26
# 29	MT1DP
# 30	hsa-mir-204
# %%
