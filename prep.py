import pandas as pd
import numpy as np
import argparse

cat_cols = ["DeviceType", "DeviceInfo"]
cat_cols += ["id_"+str(i) for i in range(12,39)]
cat_cols += ["ProductCD","addr1", "addr2", "P_emaildomain", "R_emaildomain"]
cat_cols += ["card"+str(i) for i in range(1,7)]
cat_cols += ["M"+str(i) for i in range(1,10)]
target_col = "isFraud"
cols_to_drop = ["isTrain"]


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="Specify seed for reproducibility", type=int, default=42)
parser.add_argument("-d", "--datadir", help="Path to datafiles", type=str, default="data/")
parser.add_argument("-o", "--outdir", help="Path to output datafiles", type=str, default="prepped/")


def split_dataset(dataframe, val_index, holdout_test_index):
    train = dataframe.iloc[:val_index]
    val = dataframe.iloc[val_index:holdout_test_index]
    holdout_test = dataframe.iloc[holdout_test_index:]
    return train, val, holdout_test

if __name__ == "__main__":

    args = parser.parse_args()
    print("Using arguments:")
    print(args)
    seed = args.seed
    datadir = args.datadir
    outdir = args.outdir
    print("Reading files from "+datadir)
    df_id = pd.read_csv(datadir+"train_identity.csv")
    df_t = pd.read_csv(datadir+"train_transaction.csv")
    df_t["isTrain"] = True
    df_idt = pd.read_csv(datadir+"test_identity.csv")
    df_tt = pd.read_csv(datadir+"test_transaction.csv")
    df_tt["isTrain"] = False
    df_id = pd.concat([df_id, df_idt], axis=0, sort=True)
    df_t = pd.concat([df_t, df_tt], axis=0, sort=True)
    print("Merging input files")
    df = df_t.merge(df_id, how="left", on="TransactionID")
    print("Transforming categorical columns")
    df.loc[:,cat_cols] = df.loc[:,cat_cols].apply(pd.Categorical)
    for c in cat_cols:
        df[c] = df[c].cat.codes
    df.fillna(-1, inplace=True)
    df_train = df.loc[df.isTrain]
    df_test = df.loc[~df.isTrain]
    del df
    X_train = df_train.drop([target_col]+cols_to_drop, axis=1)
    y_train = df_train[target_col]
    X_test = df_test.drop([target_col]+cols_to_drop, axis=1)
    y_test = df_test[target_col]
    print("Saving preprocessed files to "+datadir+outdir)
    X_train.to_pickle(datadir+outdir+"X_train_"+".pkl")
    y_train.to_pickle(datadir+outdir+"y_train_"+".pkl")
    X_test.to_pickle(datadir+outdir+"X_test_"+".pkl")
    y_test.to_pickle(datadir+outdir+"y_test_"+".pkl")