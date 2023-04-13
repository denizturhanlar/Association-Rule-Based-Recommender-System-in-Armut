
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih



import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.
df_ = pd.read_csv("WEEK 5 RECOMMENDATION SYSTEMS/armut_data.csv")
df = df_.copy()
df.head()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.
#######################################################################################################

# NOT:
# Service "Category" altında alınan hizmettir.
# Mesela;
# "Temizlik" kategorisi altında "Koltuk Temizligi" servistir.
# "Tamirat" kategorisinin altında "musluk tamiratı" bir servistir.


df.values

# for row in df.values = tüm satır degerlerinde gez.
# her satırın degerleri;
# row[0] = userid
# row[1] = serviceid
# row[2] = categoryid
# row[3] = createdate

df.values

df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

#--------------------------------------------------------------------------------------------------------------------------
# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.

# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin 25446 id'li müşteri 2017'in 8.ayında aldığı 4_5, 48_5, 6_7, 47_7  hizmetler bir sepeti; 2017'in 9.ayında
# aldığı 17_5, 14_7  hizmetler başka bir sepeti ifade etmektedir.

# Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini kullanıcı bazında "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
############################################################################################################################################

# öncelikle createdate değişkenini datatime türüne cevirmeliyiz.
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()

# Elimizde bir sepet veya bir fatura yok bu sebeple;
# Biz bir sepet olusturmaya calısıyoruz.
# Bu sepeti bir aylık süreyle olusturalım.
# Yani, tarih değişkenini ay ve yıl olarak bölelim.

df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df["NEW_DATE"].head()

# UserID ve yeni oluşturduğunuz date değişkenini kullanıcı bazında "_" ile birleştirirek ID adında yeni bir değişkene atayınız.

# daha sonra row[0] userid ve row[5] new_date'i birleştirerek aslında bir sepet olusturmus oluyoruz.
# aynı sepetid'ye ait olanlar bir sepette yani 1 ayda alınmıs hizmetler oluyor.
df.head()
df.values
df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()


df.sort_values("SepetID").head() # aynı müşterinin aldıgı hizmetler farklı satırlarda olmak üzere görülür.

###############################################
# GÖREV 2: Birliktelik Kuralları Üretiniz
###############################################

############################################################################################################
# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..
############################################################################################################################################

# aslında bu adımda bizden istenen item_set olusturmaktır. yani;
# satırlarda sepet olsun,
# sutun değişkenleri hizmet olsun,
# kesişimlerinde hizmet degerlerinin alınıp alınmadıgı bilgisi ifade edilsin.

# "sepetid", "hizmet"  kırılımında grupluyoruz.
# "hizmet adedi" (count) hesaplıyoruz.
# unstack ile pivotlaştırma yapıyoruz yani tam olarak istediğiz formata getiriyoruz.
# unstack -> satıra sepetleri , sutuna hizmetleri , üçüncü kırılım olan hizmet adedini kesişimlerine yani hücrelere yerleştiriyor.
# nan degerleri 0 ile dolduruyoruz (fillna(0))
# son olarak; hizmet sayısı 1'den farklı olabilir.
# biz adediyle ilgilenmiyoruz burada tek ilgilendiğimiz hizmeti alıp almamıs olması.
# "applymap(lambda x: 1 if x > 0 else 0" applymap ile tüm hücreleri gez ve eger 0'dan büyükse 1 yaz, degilse 0
# bu şekilde hücrelerde alındı alınmadı bilgisiyle doldurmus oluyoruz.

df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().head()
df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().head()
df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0)
invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()


########################################################################
# Adım 2: Birliktelik kurallarını oluşturunuz.
########################################################################

# Apriori algoritması birliktelik kuralı olusturabilmek için tabloyu hazırlar.
# min_support ister.

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
frequent_itemsets.tail()

#frequent_itemsets = fpgrowth(invoice_product_df, min_support=0.01, use_colnames=True)



# Birliktelik kurallarını olusturmalıyız.

# Kurallar lift,confidence, support degerlerini hesaplar.
# antecedents = X
# consequents = Y
# Kuralları apriori ile hazırladıgımız tablo üzerinden hesaplatıyoruz.

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

# böylece birliktelik kurallarını olusturmus oluyoruz.

################################################################################################################################################
#Adım 3: arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
################################################################################################################################################

# Tavsiye olusturalım:

# rec_count: kac ürün tavsiyesi istiyorsun?
# product_id: hangi ürü için tavsiye istiyorsun?
# rules_df: kullanacak oldugum kural listesi.


rules.sort_values("lift", ascending=False).head()

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # kuralları lifte göre büyükten kücüğe sıralar. (en uyumlu ilk ürünü yakalayabilmek için)
    # confidence'e göre de sıralanabilir insiyatife baglıdır.
    recommendation_list = [] # tavsiye edilecek ürünler için bos bir liste olusturuyoruz.
    # antecedents: X
    #items denildigi için frozenset olarak getirir. index ve hizmeti birleştirir.
    # i: index
    # product: X yani öneri isteyen hizmet
    for i, product in enumerate(sorted_rules["antecedents"]): #enumerate
        for j in list(product): # hizmetlerde(product) gez:
            if j == product_id:# eger tavsiye istenen ürün yakalanırsa:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # index bilgisini i ile tutuyordun bu index bilgisindeki consequents(Y) değerini recommendation_list'e ekle.

    # tavsiye listesinde tekrarlamayı önlemek için:
    # mesela 2'li 3'lü kombinasyonlarda aynı ürün tekrar düşmüş olabilir listeye gibi;
    # sözlük yapısının unique özelliginden yararlanıyoruz.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] # :rec_count istenen sayıya kadar tavsiye ürün getir.


# 2_0 hizmeti için 1 tane tavsiye olsuturalım:
arl_recommender(rules, "2_0", 1)


sorted_rules = rules.sort_values("lift", ascending=False).head()
sorted_rules["antecedents"].items()

