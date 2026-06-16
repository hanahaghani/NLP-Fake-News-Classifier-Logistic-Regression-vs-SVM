import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,LocallyLinearEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import LinearSVC

import joblib

data=pd.read_csv('fake_and_real_news_dataset.csv')

data.info()
print(data.describe())
print(data.head())
print(data.isnull().sum())

data=data.drop(columns='idd')
data=data.dropna()

data['content']=data['title']+' '+data['text']
x=data['content']

print(x.isnull().sum())

y=(data['label']=='REAL').astype(int)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

vectorize=TfidfVectorizer(
    stop_words='english',
    max_features=3000
)
x_train_vectorized=vectorize.fit_transform(x_train)
x_test_vectorized=vectorize.transform(x_test)

preprocessing=Pipeline([
    ('scaler',StandardScaler()),
    ('pca',PCA(n_components=2))
])
x_pca_train=preprocessing.fit_transform(x_train_vectorized.toarray())
x_pca_test=preprocessing.transform(x_test_vectorized.toarray())
plt.figure(figsize=(10,8))
y_color=np.where(y_train.values.ravel()==1,'red','blue')
scatter=plt.scatter(
    x_pca_train[:,0],x_pca_train[:,1],c=y_color,s=5
)
plt.title('pca projection')
plt.savefig('img/pca-scatter.png')
plt.show()

tsne=TSNE(
    n_components=3,perplexity=30,random_state=42,init='pca',learning_rate='auto'         
)
x_train_tsne=tsne.fit_transform(x_train_vectorized)
plt.figure(figsize=(10,8))
scatter=plt.scatter(
    x_train_tsne[:,0],x_train_tsne[:,1],c=y_color,s=5
)
plt.title('tsne projection')
plt.savefig('img/tnse-scatter.png')
plt.show()

lle=LocallyLinearEmbedding(
    n_components=2,n_neighbors=10,random_state=42
)
x_train_lle=lle.fit_transform(x_train_vectorized.toarray())
x_test_lle=lle.transform(x_test_vectorized.toarray())
plt.figure(figsize=(6,6))
plt.scatter(x_train_lle[:,0],x_train_lle[:,1],c=y_color,s=15)
plt.title('lle projection')
plt.savefig('img/lle-scatter.png')
plt.show()

model=LogisticRegression(random_state=42,max_iter=1000)
model.fit(x_pca_train,y_train)
y_prob=model.predict(x_pca_test)

print(classification_report(y_test,y_prob))
print(cross_val_score( model,x_pca_train, y_train, cv=3, scoring="accuracy"))

cm_pca=confusion_matrix(y_test,y_prob)
plt.figure(figsize=(5,4))
sns.heatmap(cm_pca ,annot=True , fmt='d',cmap='Blues',
            xticklabels=['Fake','Real'],
            yticklabels=['Fake','Real'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix logistic model & pca')
plt.savefig('img/confusion_matrix_pca.png')
plt.show()

print(model.score(x_pca_train,y_train))
print(model.score(x_pca_test,y_test))

###
model.fit(x_train_vectorized,y_train)
y_prob=model.predict(x_test_vectorized)

print(classification_report(y_test,y_prob))
print(cross_val_score( model,x_train_vectorized, y_train, cv=3, scoring="accuracy"))

cm=confusion_matrix(y_test,y_prob)
plt.figure(figsize=(5,4))
sns.heatmap(cm ,annot=True , fmt='d',cmap='Blues',
            xticklabels=['Fake','Real'],
            yticklabels=['Fake','Real'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix logistic model')
plt.savefig('img/confusion_matrix.png')
plt.show()

print(model.score(x_train_vectorized,y_train))
print(model.score(x_test_vectorized,y_test))

params={
    'C':[1.8,1,1.6,1.7,2,1.5,0.9,1.2]
}

grid=GridSearchCV(
    LinearSVC(),params,cv=5,scoring='f1'
)

grid.fit(x_train_vectorized,y_train)
print(grid.best_params_)
print(grid.best_score_)

y_prob_grid=grid.predict(x_test_vectorized)
print(classification_report(y_test,y_prob_grid))

cm=confusion_matrix(y_test,y_prob_grid)
plt.figure(figsize=(5,4))
sns.heatmap(cm ,annot=True , fmt='d',cmap='Blues',
            xticklabels=['Fake','Real'],
            yticklabels=['Fake','Real'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix linearSVC model')
plt.savefig('img/confusion_matrix_SVC.png')
plt.show()

print(grid.score(x_train_vectorized,y_train))
print(grid.score(x_test_vectorized,y_test))

new_data=x_test_vectorized[10]
y_new=grid.predict(new_data)

if y_new==0:
    print("Fake")
else:
    print("Real")

joblib.dump(grid,'model/linear_svc_model.joblib')
joblib.dump(vectorize,"model/tfidf_vektorizer.joblib")
print('model saved')