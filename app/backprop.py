from . import app,render_template,request,df,pd,request,convert
from .routes import pd
import numpy as np

df = pd.read_csv('Data_Tanaman_Padi.csv')

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 3
    self.outputSize = 1
    self.hiddenSize = 3
 
    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
 
  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o 
 
  def sigmoid(self, s):
    # activation function 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

@app.route('/backpropagation')
def backpropagation():
  return render_template('backprop.html',title='Backpropagation')

def hitung_prediksi(provinsi,suhu_rata,kelembapan,luas_panen):
  prov = df[df['Provinsi'] == provinsi]
  X_test = np.array([[luas_panen,kelembapan,suhu_rata]])
  X_train = prov[['Luas Panen','Kelembapan','Suhu rata-rata']].values
  y_train = prov[['Produksi']].values
  X_scaler = X_train/np.amax(X_train, axis=0) 
  y_scaler = y_train/max(y_train)
  X_test_scaler = X_test/np.amax(X_test,axis=0)
  
  NN = Neural_Network()
  for i in range(1000):
    NN.train(X_scaler,y_scaler)
  hasil = NN.forward(X_test_scaler)[0][0]
  return f'{convert(hasil*max(y_train)[0])} Ton / Tahun'



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
    hasil = hitung_prediksi(provinsi,float(suhu_rata),float(kelembapan),float(luas_panen))
    return render_template('prediksi.html',prov=data_prov,data_mean=dmean,hasil=hasil)
    
    
  return render_template('prediksi.html',prov=data_prov,data_mean=dmean,title='Prediksi')
