import pickle
from sklearn.preprocessing import StandardScaler as sc
sc=pickle.load(open('models/scaler.pkl','rb'))
model=pickle.load(open('models/ridge.pkl','rb'))
print('the :',sc.transform([[12,1,1,1,1,1,1,1,1]]))
print('the output:',model.predict(sc.transform([[12,1,1,1,1,1,1,1,1]])))