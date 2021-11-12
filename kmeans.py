import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

retail = pd.read_csv("sample.csv")
retail.head()
retail.info()
ritel = retail.drop(["ADDRESS", "COMPANY", "GUID", "IMAGE","LOCAL_IMAGE","MAP_BOUNDARIES","NAME","REPORT_TYPE.GUID","REPORT_TYPE.NAME","STATUS","TIMESTAMP","TYPE","UNIT","_id"], axis = 1)
# ritel = retail.drop(["ADDRESS", "COMPANY", "GUID", "IMAGE","LOCAL_IMAGE","TIMESTAMP","NAME","STATUS","TIMESTAMP","UNIT","_id"], axis = 1)
ritel.head()
ritel.info()
ritel_x = ritel.iloc[:, 1:4]
ritel_x.head()
ritel_x.info()
sns.scatterplot(x="LONG", y="LAT", data=ritel, s=100, color="pink", alpha = 0.5)
x_array = np.array(ritel_x)
# x_array = np.nanmin(ritel_x, axis=1)
print(x_array)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled
# Menentukan dan mengkonfigurasi fungsi kmeans
kmeans = KMeans(n_clusters = 5, random_state=123)
# Menentukan kluster dari data
kmeans.fit(x_scaled)
print(kmeans.cluster_centers_)
# Menampilkan hasil kluster
print(kmeans.labels_)
# Menambahkan kolom "kluster" dalam data frame ritel
ritel["kluster"] = kmeans.labels_
ritel.head()

fig, ax = plt.subplots()
sct = ax.scatter(x_scaled[:,1], x_scaled[:,0], s = 100,
c = ritel.kluster, marker = "o", alpha = 0.5)
centers = kmeans.cluster_centers_
ax.scatter(centers[:,1], centers[:,0], c='blue', s=200, alpha=0.5);
plt.title("Hasil Klustering K-Means")
plt.xlabel("Scaled Longitude")
plt.ylabel("Scaled Latitude")
plt.show()
