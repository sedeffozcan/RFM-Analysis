import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_ = pd.read_csv("/Users/sedeftaskin/Desktop/Data_Science/archive/bookings.csv")
df = df_.copy()

# The following dataset contains some categorical or numerical variables belonging to customers.
#
# These variables are:
# * GuestID': Identification Number of Guest
# * 'Status': Status of Booking
# * 'RoomGroupID': Identification Number of Room Group
# * 'CreatedDate': Date of Creating Booking
# * ArrivalDate': Date of Arrival
# * 'DepartureDate': Date of Departure
# * 'RoomPrice': Price of Room per Day
# * 'Channel': Channel of Booking
# * 'RoomNo': Identification Number of Room
# * 'Country': Country of Guest
# * 'Adults': Number of Adults
# * 'Children': Number of Children
# * 'TotalPayment': Total Fee Paid
#
# The Guest ID is not unique because a customer can make more than one reservation.
# Even a (GuestID, ArrivalDate) pair can be duplicated if the customer booked more than 1 room in a day
# (for family, a group of friends, or the entire company).
def check_df(dataframe, head=5, tail=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T)
    print(dataframe.describe(include=object).T)
    print("###################### Unique Values #################")
    print(dataframe.nunique())

check_df(df)

# Some outliers

df=df[(df["Adults"]!=0) | (df["Children"]!=0)]
df = df[df["RoomPrice"]>0]

# Missing Values
col_without_NaN = []
col_with_NaN = []
for col in df.columns:
    if any(df[col].isnull()):
        col_with_NaN.append(col)
    else:
        col_without_NaN.append(col)

# Filling them

df["Channel"].fillna("other", inplace=True)
df["Country"].fillna("otr", inplace=True)
# filling roomno median?
df.dropna(inplace=True)
#RFM metrics
#Analysis Date
df["DepartureDate"].max()
analysis_date = dt.datetime(2020, 9, 28)
#Converting to time variables
for col in df.columns[df.columns.str.contains("Date")]:
    df[col] = pd.to_datetime(df[col])

rfm = df.groupby("GuestID").agg({"DepartureDate": lambda x: (analysis_date - x.max()).days,
                                 "ArrivalDate": "nunique",
                                 "TotalPayment": "sum"})

rfm.columns = ['recency', 'frequency', 'monetary']
rfm.head()

#RFM Scores
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.head()

# Segmentation
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm.sort_values("recency").head(20)
rfm[["segment","recency", "frequency", "monetary"]].groupby("segment").agg("mean").sort_values("monetary", ascending=False)

df_target = pd.DataFrame()
df_target["Customers_At_Risk"] = rfm[rfm["segment"] == "at_Risk"].index
df_target