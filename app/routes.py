from app import app,redirect,render_template,send_from_directory,send_file,request
import pandas as pd
from babel import numbers
import numpy as np



df = pd.read_csv('Data_Tanaman_Padi.csv')
df_de = df.copy()

def convert(x):
    return numbers.format_decimal(x)

df.Produksi = df['Produksi'].apply(convert)
df['Luas Panen'] = df['Luas Panen'].apply(convert)





@app.route('/')
def home():
    title = 'Dashboard'
    return render_template('index.html',len_data=len(df),title=title)


@app.route('/table')
def table():
    return render_template('tables.html',data=df,total=len(df),title='Dataset')
    # return render_template_string(df.to_html())
@app.route('/analisis_data_provinsi/<provinsi>')
def provinsi(provinsi):
    if provinsi == 'sumut':
        data = df_de[df_de['Provinsi'] == 'Sumatera Utara']
        prov = "Sumatera Utara"
        pages = ['curah_hujan_1.html','kelembapan_1.html','suhu_rata-rata.html','luas_panen.html']
    else:
        data = df_de[df_de['Provinsi'] == 'Riau']
        prov = "Riau"
        pages =  ['curah_hujan_2.html','kelembapan_2.html','suhu_rata-rata_2.html','luas_panen_2.html'  ]
    info = {'mean_luas_panen':int(data['Luas Panen'].mean()),'max_luas_panen':int(data['Luas Panen'].max()),'min_luas_panen':int(data['Luas Panen'].min()),'mean_curah_hujan':int(data['Curah hujan'].mean()),'maks_curah_hujan':int(data['Curah hujan'].max()),'min_curah_hujan':int(data['Curah hujan'].min()),'mean_kelembapan':int(data['Kelembapan'].mean()),'min_kelembapan':int(data['Kelembapan'].min()),'maks_kelembapan':int(data['Kelembapan'].max()),'mean_suhu':int(data['Suhu rata-rata'].mean()),'maks_suhu':int(data['Suhu rata-rata'].max()),'min_suhu':int(data['Suhu rata-rata'].min())}
    return render_template('provinsi_1.html',data=data,info=info,pages=pages,title=f'Analisis Data {provinsi}',prov=prov)
    
    
@app.route('/report')
def report():
    return send_file('upload/files/report.html')

@app.route('/table/<provinsi>')
def table_provinsi(provinsi):
    pr = str(provinsi).title()
    
    if provinsi:
        if provinsi == 'total_produksi':
            provinsi = {}
            

            for p in df_de.Provinsi.unique():
                produksi = df_de[df_de['Provinsi'] == p]['Produksi']
                rata_rata = convert(np.mean(produksi))
                total_produksi = convert(sum(produksi))

                provinsi[p] = [rata_rata,total_produksi] 

            
            return render_template('tables_total_produksi.html',data=provinsi,title="Total Produksi") 
        else:
            data = df[df['Provinsi'] == pr]
            return render_template('tables.html',data=data,total=len(data),title=provinsi)
    else:
        "Provinsi Tidak ditemukan"

    
