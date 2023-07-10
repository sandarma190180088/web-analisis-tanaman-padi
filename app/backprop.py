from . import app,render_template,request,df,pd,request,convert
from .routes import pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly

df = pd.read_csv('data_lengkap.csv')

class Neural_Network(object):
  def __init__(self):
    # inisisasi nilai input layer, bobot, dan output layer
    self.inputSize = 3
    self.outputSize = 1
    self.hiddenSize = 3
 
    #bobot
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 
 
  def forward(self, X):
    self.z = np.dot(X, self.W1) 
    self.z2 = self.sigmoid(self.z) 
    self.z3 = np.dot(self.z2, self.W2) 
    o = self.sigmoid(self.z3) 
    return o 
 
  def sigmoid(self, s):
    # fungsi aktivasi sigmoid
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    # fungsi derivatif sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    self.o_error = y - o 
    self.o_delta = self.o_error*self.sigmoidPrime(o) 

    self.z2_error = self.o_delta.dot(self.W2.T) 
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) 
    
    # update nilai bobot
    self.W1 += X.T.dot(self.z2_delta)
    self.W2 += self.z2.T.dot(self.o_delta)

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

@app.route('/backpropagation')
def backpropagation():
  return render_template('backprop.html',title='Backpropagation')

def hitung_prediksi(provinsi,X):
  prov = df[df['Provinsi'] == provinsi]
  X_test = X
  X_train = prov[['Luas Panen','Kelembapan','Suhu rata-rata']].values
  y_train = prov[['Produksi']].values
  X_scaler = X_train/np.amax(X_train, axis=0) 
  y_scaler = y_train/max(y_train)
  X_test_scaler = X_test/np.amax(X_test,axis=0)
  
  NN = Neural_Network()
  for i in range(1000):
    NN.train(X_scaler,y_scaler)
  hasil = NN.forward(X_test_scaler)
  
  return hasil*max(y_train)


@app.route('/prediksitahun',methods=['GET','POST'])
def prediksi_():
    if request.method == 'POST':
        provinsi = request.form['provinsi']
        X = np.random.rand(4,3)
        hasil = hitung_prediksi(provinsi,X)
        X = ['2021','2022','2023','2024']
        hasil = [i[0] for i in hasil]
        hasil = pd.DataFrame({'Tahun':X,'Prediksi':hasil})

        fig = px.line(x = hasil['Tahun'], y =hasil['Prediksi'], template='plotly_dark', markers=True)
        fig.update_layout(title=f'Prediksi Produksi di Provinsi {provinsi}  Dari Tahun 2021-2024',
                   xaxis_title='Tahun',
                   yaxis_title='Jumlah Produksi')
        return fig.to_html()

    prov = df["Provinsi"].unique()
    return render_template('prediksi_.html',prov=prov,title='Prediksi 4 Tahun')

@app.route('/prediksi',methods=['GET','POST'])
def prediksi():
  data_prov = df.Provinsi.unique()
  dmean = df[['Luas Panen','Kelembapan','Suhu rata-rata']].mean().round()
  # data_mean = {'luas_panen':dmean[0],'kelembapan':dmean[1],'suhu':dmean[2]}
  
  if request.method == 'POST':
    provinsi = request.form['provinsi']
    luas_panen = request.form['luas_panen']
    kelembapan = request.form['kelembapan']
    suhu_rata = request.form['suhu_rata']

    X_test = np.array([[float(luas_panen),float(kelembapan),float(suhu_rata)]])
    hasil = hitung_prediksi(provinsi,X_test)
    hasil = f'{convert(hasil[0][0])} Kg / {convert(hasil[0][0] / 1000)} Ton'
    return render_template('prediksi.html',prov=data_prov,data_mean=dmean,hasil=hasil)
    
    
  return render_template('prediksi.html',prov=data_prov,data_mean=dmean,title='Prediksi')
