import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Değişken Özellikleri

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


# Görev 1: Veriyi Anlama ve Hazırlama

# Adım1: flo_data_20K.csv verisini okuyunuz. Dataframe’in kopyasını oluşturunuz.

df_=pd.read_csv("/Users/sedeftaskin/Desktop/Data_Science/VBO/Homeworks/WEEK3/FLO_RFM_Analizi/flo_data_20k.csv")
df=df_.copy()
df.head()
# Adım2:Veri setinde
# a. İlk 10 gözlem,
# b. Değişken isimleri,
# c. Betimsel istatistik,
# d. Boş değer,
# e. Değişken tipleri, incelemesi yapınız.

df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.info()

#Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["total_order_number"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]

df["total_price"]=df["customer_value_total_ever_online"]+df["customer_value_total_ever_offline"]
df.head()

# Adım 4: Değişken tiplerini inceleyiniz.
# Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes
#to_datatime ile
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
df["first_order_date"]=pd.to_datetime(df["first_order_date"])
df["last_order_date"]=pd.to_datetime(df["last_order_date"])
#astype ile
df['first_order_date'] = df['first_order_date'].astype('datetime64[ns]')
df.dtypes

# Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına
# bakınız.

# order channel
df['order_channel'].value_counts().plot(kind='bar')
plt.show()
sns.countplot(df["order_channel"])
df["order_channel"].value_counts()

# total_price
plt.boxplot(df["total_price"])

sns.histplot(df["total_price"])

#ürün sayısı
width_length=df["total_order_number"].sum()/df["total_order_number"].count()
width_length
df["total_order_number"].nunique()
df["total_order_number"].max()
df["total_order_number"].median()
df["total_order_number"].mean()
sns.histplot(df["total_order_number"],bins=[0,5,10,15,20,25,30,35,40,50,55,60,65,70,75,df["total_order_number"].max()])

df.groupby("order_channel").agg({"total_price": "sum",
                                 "total_order_number":"sum",
                                 "master_id": "count"})

# Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız

df.sort_values(by="total_price",ascending=False).head(10)
df.groupby("master_id").agg({"total_price": "sum"}).sort_values("total_price", ascending=False).head(10)
# Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.sort_values(by="total_order_number",ascending=False).head(10)
df.groupby("master_id").agg({"total_order_number": "sum"}).sort_values("total_order_number", ascending=False).head(10)
# Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.
# read ile dene
def data_prep(df):

    print(df.head(10))
    print(df.columns)
    print(df.describe().T)
    print(df.isnull().sum())
    print(df.info())

    df["total_order_number"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

    df["total_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

    print(df.dtypes)
    df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
    df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
    df["first_order_date"] = pd.to_datetime(df["first_order_date"])
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])

    print(df.groupby("order_channel").agg({"total_price": "sum",
                                     "total_order_number": "sum",
                                     "master_id": "count"}))

    print(df.sort_values(by="total_price", ascending=False).head(10))
    print(df.sort_values(by="total_order_number", ascending=False).head(10))

    return df

data_prep(df)
# Görev 2: RFM Metriklerinin Hesaplanması

# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.

df["last_order_date"].max()
analysis_date = dt.datetime(2021,5,31)
rfm = df.groupby('master_id').agg({'last_order_date': lambda date : (analysis_date - date.max()).days,
                                     'total_order_number': "sum",
                                     'total_price':"sum"})

rfm.columns = ['recency', 'frequency', 'monetary']
rfm.head()
# Görev 3: RF Skorunun Hesaplanması
# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))
rfm.head()

rfm.describe().T
rfm[rfm["RF_SCORE"] == "55"]

# Görev 4: RF Skorunun Segment Olarak Tanımlanması
# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
# Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

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
# regex cheatsheet-- bak

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm.head()

# Görev 5: Aksiyon Zamanı !

# Adım 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

# Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde.
# Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçmek isteniliyor.
# Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

df.head()
rfm.head()
yeni_df=df.merge(rfm,on="master_id", how="left")
yeni_df.head()
yeni_df_a=yeni_df.loc[((yeni_df["segment"]=="champions") | (yeni_df["segment"]=="loyal_customers")) & (yeni_df["interested_in_categories_12"].str.contains("KADIN"))]
yeni_df_a.to_csv("champions,loyal_customers,KADIN.csv")

# b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır.
# Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler,
# uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor.
# Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.


yeni_df_b=yeni_df.loc[((yeni_df["segment"]=="cant_loose") | (yeni_df["segment"]=="new_customers")|(yeni_df["segment"]=="hibernating")) & (yeni_df["interested_in_categories_12"].str.contains("ERKEK") | (yeni_df["interested_in_categories_12"].str.contains("COCUK")))]
yeni_df_b.to_csv("champions,loyal_customers,KADIN.csv")

yeni_df_b.head(15)






