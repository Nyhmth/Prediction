import numpy as np
from sklearn.linear_model import LinearRegression

#Database
# x = Data, y = Target
x = [[1],[3],[5],[7],[9],[11],[13],[15],[17],[19], [21]]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20,22]

regr = LinearRegression().fit(x,y)
regr.score(x, y)



#Data uji
predict = np.array([[23]])

#Menampilkan data prediksi
print("Prediksi")
print("Input = ", predict)
print("Output = ", regr.predict(predict))

