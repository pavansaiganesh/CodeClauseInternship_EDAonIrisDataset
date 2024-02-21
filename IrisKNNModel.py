import joblib 

knn = joblib.load('filename.pkl') 
arr = knn.predict([[7.6,3,6.6,2.0],
                     [6.7,3.1,4.4,1.4]])
for i in arr:
    if i == 0:
        print('Iris Setosa')
    elif i == 1:
        print('Iris Versicolor')
    else:
        print('Iris Virginica')