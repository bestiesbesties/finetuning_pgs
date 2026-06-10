# import os
# from importlib import reload
# import pandas as pd
# from src.sampeling import confidence, histing
# from src._config import config

# SECOND_EVALUATION_SEED = 200
# SECOND_EVALUATION_SIZE = 2000
# DATA_NAME = F"SE_{SECOND_EVALUATION_SEED}_{SECOND_EVALUATION_SIZE}"

# # dfo = pd.read_csv(os.path.join("data", "parsed-pgs33_pruned.csv"))
# # df = undersampling.undersample(dfo, SECOND_EVALUATION_SIZE, SECOND_EVALUATION_SEED)
# # df_train, df_test = train_test.split(df, SECOND_EVALUATION_SEED)
# # df_test.to_csv(os.path.join("data", DATA_NAME + "_test.csv"), index=False)
# # df_train.to_csv(os.path.join("data", DATA_NAME + "_train.csv"), index=False)


# pool = pd.read_csv(os.path.join("data", DATA_NAME + "_train.csv"))
# results = pool["text"].apply(confidence.route)
# pool["conf"] = results.str[0]
# pool["pred"] = results.str[1]
# pool.to_csv(os.path.join("data", DATA_NAME + "_pool.csv"), index=False)


# hist = histing.stratified_sample(pool, 50)
# hist.to_csv(os.path.join("data", DATA_NAME + "_hist.csv"), index=False)



# # print(confidence.route(pool.iloc[1398]["text"]))

# pool.sort_values("conf")

# df = pool
# N = 50

