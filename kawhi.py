#import libraries
import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
from sklearn import manifold,ensemble 
from sklearn.decomposition import TruncatedSVD 
import math 

nba_peach=pd.read_csv("peach_nba.csv")
nba_peach1=nba_peach.iloc[:,5:28]
nba_peach1=nba_peach1.dropna()

#t-sne (reduce dataset dimensionality)
X_reduced=TruncatedSVD(n_components=10,random_state=0).fit_transform(age_25_sub)
tsne=manifold.TSNE(n_components=2,perplexity=30,verbose=2)
X_tsne=tsne.fit_transform(X_reduced)

#KNN 
age_25=nba_peach[nba_peach["Age"]==25].dropna()
age_25_vorp=age_25['VORP'].sort_values()
vorp_25=age_25[['Player','VORP']]

kawhi=age_25[age_25['Player']=="Kawhi Leonard"].iloc[0] #kawhi 16-17
kawhi=age_25[age_25['Player']=="Kawhi Leonard"].iloc[0]
kawhi_sub=kawhi[['VORP','FG.','X2P.','X3P.','eFG.','FT.','TS.','mpg',
'fg','fga','tfg','tfga','three_fga','fta','ord','drb','trb','ast','stl',
'blk','tov','pf','pts']]

#features used in the knn 
distance_columns=['VORP','FG.','X2P.','X3P.','eFG.','FT.','TS.','mpg',
'fg','fga','tfg','tfga','three_fga','fta','ord','drb','trb','ast','stl',
'blk','tov','pf','pts']

#euclidean distance function 
def euclidean_distance(row):
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - kawhi[k]) ** 2
    return math.sqrt(inner_value)

#players similar to Kawhi's 16-17 season 
kawhi_scores=age_25_sub.apply(euclidean_distance,axis=1)
kawhi_scores1=pd.DataFrame(kawhi_scores,columns=['distance'])
kawhi_distance_df=pd.concat([age_25,kawhi_scores1],axis=1)
kawhi_sub_distance=kawhi_distance_df[["Player","distance"]]
kawhi_sub_distance.sort_values('distance')

#convert to csv 
kawhi_sub_distance.to_csv("kawhi_sim.csv")


