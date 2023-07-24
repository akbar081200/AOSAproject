import pandas as pd                                       # manipulasi dataframe
import numpy as np                                        # manipuasi array
import matplotlib.pyplot as plt                           # visualisasi
from matplotlib.pyplot import figure                      # mengatur figur (ukuran, dll)
from wordcloud import WordCloud                           # visualisasi kata
import string                                             
import nltk
import re
import tqdm
import pyLDAvis
import pyLDAvis.gensim_models
import gensim
from nltk.tokenize import word_tokenize
from gensim.models import CoherenceModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from io import BytesIO
import base64

factory = StemmerFactory()
stemmer = factory.create_stemmer()
punctuation = string.punctuation

negasi = ['bukan','tidak','gk','ga','gak','kurang','belum']
lexicon = pd.read_csv('D:\\skripsi\\projekskripsi\\webflask\\webapp\\application\\static\\full_lexicon.csv')

stopwords_raw = nltk.corpus.stopwords.words("indonesian")
stopwords = [i for i in stopwords_raw if i not in list(lexicon['word'])]
stopwords.remove('belum')

new_stopwords = ['yg','ya','iya','sy','saya','aku','nya','ada','yang','sangat', 'menurut','sudah','dalam','dari'
                 ,'karena','cukup','pada','seperti','dapat','itu']
stopwords.extend(new_stopwords)

def read_data(data_file):                                     # input fungsi berupa file
    kolom_df = ['timestamp', 'ulasan', 'rating']              # membuat nama kolom baru
    df_raw = pd.read_csv(data_file)                           # membaca file menjadi dataframe
    for i in range(len(df_raw.columns.values)):               # mengubah nama kolom 
        df_raw.columns.values[i] = kolom_df[i]
    df_raw.drop_duplicates(subset=['ulasan'], inplace = True) # menghapus duplikat
    df_raw.reset_index(drop=True, inplace = True)             # reset index
    return df_raw

# text preprocessing
def data_preprocessing(data):

    data['s_splitting'] = [split_teks(i) for i in data.ulasan]                   # mengubah paragraf menjadi kalimat untuk nantinya dihitung nilai polaritas sentimennya    
    data['case_folding'] = [[i.lower() for i in j] for j in data.s_splitting]    # Mengubah dokumen menjadi format kasus yang seragam (huruf kecil)
    data['r_punctuation'] = [menghapus_tanda_baca(i) for i in data.case_folding] # menghilangkan tanda baca yang tidak diperlukan
    data['r_stopwords'] = [menghapus_stopwords(i) for i in data.r_punctuation]   # berfokus hanya pada kata yang memiliki makna penting   
    data['stemming'] = [stemming(i) for i in data.r_stopwords]                   # memadatkan informasi dengan mengubah kata ke bentuk akarnya
    stopwords.extend(negasi)                                                     
    data['stemming_ec'] = [menghapus_stopwords_tambahan(i, stopwords) for i in data.stemming] # menghapus stopwords tambahan yaitu kata negasi
    data['t_kata'] = tokenisasi([' '.join(i) for i in data.stemming_ec])           # membuat kolom dengan list of list hasil tokenization

    return data

# sentence splitting
def split_teks(teks):           # input string kalimat
    teks = re.sub('[.]+', '.', teks) # mengubah tanda titik lebih dari 1 (.., ..., n) menjadi hanya 1 tanda titik (.)
    teks_final = []
    teks = teks.split('.')           # memisahkan kalimat berdasarkan tanda titik
    for i in teks:                   # perulangan terhadap kata
        i = i.lstrip()               # menghapus spasi yang tidak digunakan pada sebelah kiri teks          
        if len(i) > 10:               # hanya mengambil string dengan panjang karakter kurang dari 10
            teks_final.append(i)
    return teks_final                # mengembalikan list kalimat

# remove punctuation
def menghapus_tanda_baca(teks):              # input teks berupa string 
    tanda_baca = string.punctuation          # inisiasi tanda baca
    tanda_baca = tanda_baca.replace("-", "") # mengecualikan tanda hubung
    tanda_baca = tanda_baca.replace(",", "") # mengecualikan koma, handling untuk koma ada dibawah
    tanda_baca = tanda_baca.replace(".", "") # mengecualikan tanda titik yang nantinya akan berguna pada tahapan selanjutnya
    pola = r"[{}]".format(tanda_baca)        # membuat pola. 
                                             # r'' atau raw string literal sama seperti string pada umumnya 
                                             # akan tetapi backslash tidak akan menjadi escape character
                                             # {} adalah placeholder yang akan diisi oleh tanda_baca
                                             # [] adalah literal characters, jadi tidak terbaca
                                             # intinya adalah membuat pola yang nantinya akan dihapus
                                             
    list_kalimat = []                        # list sementara
    for i in teks:     
        i = i.replace('\n',' ')             # pembersihan tambahan menghapus baris baru dalam teks
        i = i.replace('\t',' ')             # pembersihan tambahan menghapus tab dalam teks

        i = i.replace('tangibles',' ')           # menghapus kata aspek, berfungsi agar kata tidak muncul di table dashboard
        i = i.replace('reliability',' ')         # menghapus kata aspek
        i = i.replace('responsiveness',' ')      # menghapus kata aspek
        i = i.replace('assurance',' ')           # menghapus kata aspek
        i = i.replace('emphaty',' ')             # menghapus kata aspek

        i = i.replace(',',' ')              # pembersihan tambahan menghapus koma menggantikannya dengan spasi
        i = re.sub(pola, " ", i)            # menerapkan penghapusan tanda baca pada teks
        kalimat = re.sub(' +', ' ', i)      # pembersihan tambahan menghapus spasi yang berlebih
        kalimat = re.sub('-+', '-', i)      # pembersihan tambahan menghapus tanda hubung yang berlebih
        list_kalimat.append(kalimat)
    return list_kalimat

# remove stopwords
def menghapus_stopwords(list_kalimat):
    list_kalimat_baru = []
    for kalimat in list_kalimat:
        list_kata_bersih = []
        for kata in kalimat.split():                         # perulangan untuk akses per kata
            if len(kata) > 1:                                # hanya mengambil kata dengan jumlah karakter lebih dari 1
                if kata not in stopwords:                    # cek apakah kata tidak termasuk stopwords
                    list_kata_bersih.append(kata)            # jika tidak, kata akan ditambahkan ke list
        list_kalimat_baru.append(" ".join(list_kata_bersih)) # menggabungkan list dengan pemisah spasi
                
    return list_kalimat_baru

# stemming
def stemming(list_kalimat):
    a = []
    for kalimat in list_kalimat:                  # parameter berupa list kalimat, dilakukan perulangan untuk akses per kalimat
        c = []
        for kata in kalimat.split():              # dilakukan perulangan pada kalimat untuk akses per kata
            if kata not in list(lexicon['word']): # cek apakah kata ada di dalam leksikon sentimen, jika tidak maka
                b = stemmer.stem(kata)            # kata dikembalikan ke bentuk asalnya, dengan harapan akan terdapat dalam leksikon sentimen
                c.append(b)                             
            elif kata in list(lexicon['word']):   # jika kata terdapat pada leksikon, maka tidak perlu dilakukan stemming
                c.append(kata)
            d = ' '.join(c)                       # menggabungkan list kata menjadi kalimat
        a.append(d)                               # menggabungkan kalimat-kalimat
    return a                                      # output akhir berupa list kalimat yang sudah dilakukan stemming

# extra cleaning
def menghapus_stopwords_tambahan(list_kalimat, stopwords_baru):
    list_kalimat_bersih = []
    for kalimat in list_kalimat:                     # perulangan untuk akses per kalimat
        list_kata_bersih = []
        for kata in kalimat.split():                 # perulangan untuk akses per kata
            if kata not in stopwords_baru:           # cek apakah kata tidak termasuk stopwords tambahan yaitu kata negasi
                list_kata_bersih.append(kata)
            gabung_kata = ' '.join(list_kata_bersih) # menggabungkan list kata menjadi kalimat
        list_kalimat_bersih.append(gabung_kata)      # menggabungkan kalimat-kalimat kembali dalam satu list
                
    return list_kalimat_bersih

# tokenization
def tokenisasi(list_kalimat):                                 # input berupa list kalimat hasil stemming
    list_token_kata = []
    for kalimat in list_kalimat:                                    # dilakukan perulangan pada list kalimat untuk akses per kalimat
        token = gensim.utils.simple_preprocess(kalimat, deacc=True) # dilakukan proses tokenisasi dgn syarat kata > 1 karakter
        list_token_kata.append(token)                               
    return list_token_kata                                           # hasil akhir akan berupa list of token

def optimasi_model(corpus,id2word, df):

    # pengujian
    # topics_range = [i for i in range (4,9)]
    # alpha = list(np.arange(0.01, 1, 0.3).round(2)) # Alpha parameter
    # alpha.extend(('symmetric','asymmetric'))
    # beta = list(np.arange(0.01, 1, 0.3).round(2)) # Beta parameter
    # beta.append('symmetric')

    # tes
    topics_range = [8]
    alpha = ['asymmetric'] # Alpha parameter
    beta = [0.91] # Beta parameter

    model_results = {
                    'Topics': [],
                    'Alpha': [],
                    'Beta': [],
                    'Coherence': []
                    }
    total = len(topics_range)*len(alpha)*len(beta)
    progress_bar = tqdm.tqdm(total=total)
    for k in topics_range:
        for a in alpha:
            for b in beta:
                cv = generasi_nilai_coherence(corpus=corpus, dictionary=id2word, k=k, a=a, b=b, data_token=df['t_kata'])
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)
                progress_bar.update(1)
    progress_bar.close()
    df_coherence = pd.DataFrame(model_results)
    df_coherence.sort_values(by=['Coherence'], ascending=False, inplace=True)
    df_coherence.to_csv('static/df_coherence.csv', index=False)
    
    best_model, selected_k, selected_a, selected_b = pemodelan_lda(  
            corpus=corpus,
            dictionary=id2word, 
            k=df_coherence.iloc[0]['Topics'], 
            a=df_coherence.iloc[0]['Alpha'], 
            b=df_coherence.iloc[0]['Beta']
            )

    return best_model, int(df_coherence.iloc[0]['Topics']), df_coherence, topics_range, selected_k, selected_a , selected_b

def generasi_nilai_coherence(corpus, dictionary, k, a, b, data_token):
    
    model_lda = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    coherence_model_lda = CoherenceModel(model=model_lda, texts=data_token, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

def pemodelan_lda(corpus, dictionary, k, a, b):
    
    model_lda = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    return model_lda, k, a, b

def best_topic_visualization(df,topics_range):
    x_top_topic = topics_range                                 # jumlah topik
    y_top_topic = df.groupby(['Topics'])['Coherence'].max()    # nilai coherence tertinggi pada masing-masing jumlah topik
    plt.figure()
    plt.plot(x_top_topic, y_top_topic, marker='o')
    for x,y in zip(x_top_topic,y_top_topic):
        label = "{:.3f}".format(y)
        plt.annotate(label, # this is the text
                (x,y), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(5,5), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center

    plt.xlabel(f"Jumlah Topik")
    plt.ylabel(f"Nilai Coherence")
    plt.title("Perbandingan nilai coherence score berdasarkan jumlah topik")
    # plt.show()
    bt_viz = BytesIO()
    plt.savefig(bt_viz, format='png')
    bt_viz.seek(0)
    bt_png = "data:image/png;base64,"
    bt_png += base64.b64encode(bt_viz.getvalue()).decode('utf-8')

    return bt_png

def word_cloud_topic(model_lda):
    list_img = []
    for t in range(model_lda.num_topics):
        plt.figure()                                                                                 # buat figur
        plt.imshow(WordCloud(background_color='white').fit_words(dict(model_lda.show_topic(t, 10)))) # buat wordcloud dgn 10 kata pd setiap topik
        plt.axis("off")
        plt.title("Topic #" + str(t+1))
        img_temp = BytesIO()                                              # }
        plt.savefig(img_temp, format='png')                               # serangkaian fungsi penyimpanan gambar
        img_png = "data:image/png;base64,"                                # grafik matpotlib dalam flask
        img_png += base64.b64encode(img_temp.getvalue()).decode('utf-8')  # }
        list_img.append(img_png)
    
    return list_img

def pyldavis(lda_model, corpus):
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    pyLDAvis.save_html(vis, 'static/lda.html')

def topic_aspect_mapping(model_lda, aspek):
    topik = get_topics(model_lda)            # mendapatkan topik dalam bentuk dataframe
    aspek_list = ['tangibles','reliability','assurance','responsiveness','emphaty']
    aspek_dict = {}
    for i in range(len(aspek)):
        if aspek[i] in aspek_list:
            aspek_dict[aspek[i]] = []        # deklarasi list kosong dari aspek, untuk diisi oleh keyword 

    for i in range(len(aspek)):              # perulangan sebanyak jumlah topik terbaik 0,1,2
        for j in topik[topik.columns[i]]:    # perulangan untuk akses kata/row dalam dataframe topik
            if aspek[i] in list(aspek_dict.keys()): # lakukan fungsi berikutnya apabila aspek value input ada dalam keys jadi kalau dia kosong '' itu tidak melakukan fungsi berikutnya
                aspek_dict[aspek[i]].append(j)      # akses key aspek dict sesuai urutan aspek dari form, lalu tambah topic keyword
                                                    # contoh hasil : {'tangibles':['fakultas','ruang']} dst
    return aspek_dict

def get_topics(ldamodel):
    topic_dict = {}
    for i in range(ldamodel.num_topics):
        topic_dict[f'topik{i+1}'] = list(dict(ldamodel.show_topic(i,10)).keys())

    keywords_df = pd.DataFrame(topic_dict)                   # menyimpan topik dalam bentuk datafram
    keywords_df_t = keywords_df.transpose()                  # transpose dataframe
    keywords_df_t.to_csv('static/topics_t.csv', index=False) # export dataframe dalam bentuk csv

    return keywords_df

def comment_aspect_mapping(df, aspek_dict, aspek):
    review_aspect ={}
    for i in range(len(aspek)):
        review_aspect[aspek[i]] = []
    
    print(f'aspek dalam fungsi {aspek}')
    print(f'ini apa ? = {review_aspect}')

    for i in review_aspect.keys():                              # akses setiap aspek dalam dictionary
        for j in df['stemming']:                                # akses setiap baris dalam corpus
            for k in j:                                         # akses setiap kalimat dalam baris
                if set(aspek_dict[i]).intersection(k.split()):  # jika ada salah satu kata 'topik' dalam aspek_dict
                                                                # yang termasuk dalam kalimat reviw maka
                    if k not in review_aspect[i]:               # jika review belum ada dalam list, review masuk list
                        review_aspect[i].append(k)              
                    elif k in review_aspect[i]:                 # jika sudah ada reviewnya, maka tidak melakukan apa apa
                        pass

    return review_aspect

def topic_aspect_dataframe(df, review_aspect_dict, aspek):
    kolom_aspek = {}
    for i in range(len(aspek)):
        kolom_aspek[aspek[i]] = []        # dictionary untuk kolom aspek

    df_komentar = df[['stemming','rating']].copy()
    df_komentar.rename(columns = {'stemming': 'ulasan'}, inplace=True)
    df_cpa = df_komentar.explode('ulasan').reset_index(drop=True)  # df commentar per aspect
                                                    # explode memecah list komentar-komentar menjadi hanya 1 komentar per baris

    column_k = []                           # menyimpan semua baris kategori dalam 1 kolom
    for i in df_cpa.ulasan:                 # akses setiap baris dalam kolom dataframe review
        kategori = []                       # menyimpan semua kategori dalam 1 baris
        for j in list(review_aspect_dict):  # akses setiap aspek dalam dictionary
            for k in review_aspect_dict[j]: # akses setiap review dalam aspek 
                if k == i:                  # jika review sama dengan baris dalam kolom dataframe review
                    kategori.append(j)      # simpan kategori terkait

        if len(kategori) > 0:
            column_k.append(kategori)           # simpan kategori-kategori terkait
        else:
            column_k.append(['umum'])

    for i in kolom_aspek.keys():                # perulangan keys dictionary
        kolom_final = aspek_k(column_k, i)      # memanggil fungsi aspek_k()
        df_cpa[i] = pd.Series(kolom_final)      # menambah kolom baru berisi kemunculan aspek

    column_k_series = pd.Series(column_k)                # diubah ke series terlebih dahulu agar menjadi 1 dimensi
    df_cpa['kategori'] = pd.DataFrame(column_k_series)                       # kolom kategori berbentuk list
    df_cpa['kategori'] = [', '.join(map(str, l)) for l in df_cpa['kategori']] # kolom kategori berbentuk string
    
    return df_cpa

def aspek_k(list_kategori, dk):          # dictionary key
    kolom = []
    for i in list_kategori:
        if dk in i:
            kolom.append(1)
        else:
            kolom.append(0)
    return kolom

# function to write the word's sentiment if it is founded
def found_word(ind,words,word,sen,sencol,sentiment,add,lexicon_word):
    if word in sencol:                 # jika kata yang sedang diakses ada didalam list sencol
        sen[sencol.index(word)] += 1   # tambah nilai kemunculannya dalam matrix
    else:                              # jika tidak
        sencol.append(word)            # tambahkan kata baru dalam daftar kolom
        sen.append(1)                  # sen adalah list, dan ditambah 1
        add += 1                       # add juga ditambah 1
    
    if (words[ind-1] in negasi):       # jika ada kata negasi, maka nilai sentimennya adalah sebaliknya
        sentiment += -lexicon['weight'][lexicon_word.index(word)]
    else:
        sentiment += lexicon['weight'][lexicon_word.index(word)]
    
    return sen,sencol,sentiment,add

def sentiment_polarity_calculation(df_tam,lexicon_word):
    
    sencol = []                        # kolom sentimen, menampung list kata yang akan dijadikan kolom
    senrow =np.array([])               # baris sentimen, 
    nsen = 0
    sentiment_list = []                # menampung overall sentimen setiap baris
    df_tam = df_tam[df_tam['ulasan'].notna()].reset_index(drop=True)
    
    # checking every words, if they are appear in the lexicon, and then calculate their sentiment if they do
    for i in range(len(df_tam)):                   # perulangan sebanyak jumlah ulasan

        nsen = senrow.shape[0]                     # shape adalah (baris*kolom), sehingga shape[0] berarti akses baris keberapa
        words = word_tokenize(df_tam['ulasan'][i]) # kata-kata yang terdapat dalam baris i dalam corpus ulasan
        sentiment = 0                              # overall sentimen
        add = 0                                    # parameter untuk apakah matriks akan ditambah atau tidak.
        n_words = len(words)                       # banyaknya kata dalam baris ke-i

        if len(sencol)>0:                          # jika jumlah kolom lebih dari 0 maka
            sen =[0 for j in range(len(sencol))]   # membuat list array dengan nilai 0, sebanyak jumlah kolom
        else:                                      # jika jumlah kolom masih 0
            sen =[]                                # buat list kosong

        for word in words:                                     # perulangan setiap kata dalam daftar kata-kata
            ind = words.index(word)                            # mengambil indeks dari kata yang sedang dilakukan perulangan

            if word in lexicon_word :                          # jika kata ada di dalam leksikon maka jalankan fungsi
                sen,sencol,sentiment,add= found_word(ind,words,word,sen,sencol,sentiment,add,lexicon_word)
            else:
            # jika tidak dalam kamus, coba kombinasi dengan kata sebelumnya
                if(n_words>1):
                    if ind-1>-1:
                        back_1    = words[ind-1]+' '+word
                        if (back_1 in lexicon_word):
                            sen,sencol,sentiment,add= found_word(ind,words,back_1,sen,sencol,sentiment,add,lexicon_word)
                        elif(ind-2>-1):
                            back_2    = words[ind-2]+' '+back_1
                            if back_2 in lexicon_word:
                                sen,sencol,sentiment,add= found_word(ind,words,back_2,sen,sencol,sentiment,add,lexicon_word)

        if add>0:     # jika kata baru ditemukan, perluas matriks
            if i>0:
                if (nsen == 0):
                    senrow = np.zeros([i,add],dtype=int)
                elif(i != nsen):
                    padding_h = np.zeros([nsen,add],dtype=int)
                    senrow = np.hstack((senrow,padding_h))
                    padding_v = np.zeros([(i-nsen),senrow.shape[1]],dtype=int)
                    senrow = np.vstack((senrow,padding_v))
                else:
                    padding =np.zeros([nsen,add],dtype=int)
                    senrow = np.hstack((senrow,padding))
                senrow = np.vstack((senrow,sen))


            if i==0:
                senrow = np.array(sen).reshape(1,len(sen))

        # if there isn't then just update the old matrix
        elif(nsen>0):
            senrow = np.vstack((senrow,sen))

        sentiment_list.append(sentiment)
        
    sencol.append('sentiment')
    sentiment_array = np.array(sentiment_list).reshape(senrow.shape[0],1)
    sentiment_data = np.hstack((senrow,sentiment_array))
    df_sen = pd.DataFrame(sentiment_data,columns = sencol)
    df_sen.to_csv('static/df_s_calc.csv', index=False)

    polaritas = []
    for i in df_sen.sentiment:
        if i < 0:
            polaritas.append('negative')
        elif i == 0:
            polaritas.append('netral')
        elif i > 0:
            polaritas.append('positive')

    df_tam['sentiment'] = df_sen['sentiment'].copy()
    df_tam['polaritas'] = polaritas
    
    return df_tam

def horizontal_bar_topic_per_aspect(aspek_dict,df_final_raw):
    jumlah_ulasan = []
    for i in aspek_dict.keys(): # perulangan terhadap semua aspek
        jumlah_ulasan.append(df_final_raw[df_final_raw[i]==1].shape[0]) # hitung jumlah baris dalam aspek tertentu, lalu tambah ke list

    df_barh = pd.DataFrame({
        'Aspek' : [i for i in aspek_dict.keys()],
        'Jumlah' : jumlah_ulasan
    })
    df_barh.sort_values(by='Jumlah', inplace=True)

    fig, ax = plt.subplots()

    for index, value in enumerate(df_barh['Jumlah']):
        ax.text(value, index, str(value))
        
    # ----
    ax.set_axisbelow(True)
    ax.grid(axis = "x", color="#A8BAC4", lw=1.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_lw(1.5)    
    ax.barh(df_barh['Aspek'], df_barh['Jumlah'], color=['#0fcccfff', '#ff9718ff', '#ff5ac0ff', '#b066faff', '#0189ffff'])
    plt.xlabel('Jumlah Ulasan')
    # ----
    bar_temp = BytesIO()
    plt.savefig(bar_temp, bbox_inches='tight', format='png')
    bar_temp.seek(0)
    bar_png = "data:image/png;base64,"
    bar_png += base64.b64encode(bar_temp.getvalue()).decode('utf-8')

    return bar_png
    
def stacked_bar_sentimen_per_aspect(df_final, aspek):
    # inisiasi dictionary
    banyak_sentiment_per_aspek = {}
    for i in range(len(aspek)):
        banyak_sentiment_per_aspek[aspek[i]] = []                       

    for i in banyak_sentiment_per_aspek:    # membuat list banyak sentimen positif, netral, negatif
        banyak_sentiment_per_aspek[i].append(list(df_final[df_final[i] == 1]['polaritas']).count('positive'))
        banyak_sentiment_per_aspek[i].append(list(df_final[df_final[i] == 1]['polaritas']).count('netral'))
        banyak_sentiment_per_aspek[i].append(list(df_final[df_final[i] == 1]['polaritas']).count('negative'))

    sentiment_tranpose = {                  # membuat sentimen sebagai kolomnya
            'positive' : [banyak_sentiment_per_aspek[i][0] for i in banyak_sentiment_per_aspek.keys()],
            'netral' : [banyak_sentiment_per_aspek[i][1] for i in banyak_sentiment_per_aspek.keys()],
            'negatif' : [banyak_sentiment_per_aspek[i][2] for i in banyak_sentiment_per_aspek.keys()],
            }

    r = [i for i in range(len(banyak_sentiment_per_aspek))]
    df_sentiment_tranpose = pd.DataFrame(sentiment_tranpose)

    totals = [i+j+k for i,j,k in zip(df_sentiment_tranpose['positive'], 
                                     df_sentiment_tranpose['netral'], 
                                     df_sentiment_tranpose['negatif'])]
    
    positiveBars = [round(i / j,3) * 100 for i,j in zip(df_sentiment_tranpose['positive'], totals)]
    netralBars = [round(i / j,3) * 100 for i,j in zip(df_sentiment_tranpose['netral'], totals)]
    negatifBars = [round(i / j,3) * 100 for i,j in zip(df_sentiment_tranpose['negatif'], totals)]

    # plot
    barWidth = 0.75
    names = [i for i in banyak_sentiment_per_aspek.keys()]
    
    fig, ax = plt.subplots()
    # Create green Bars
    p1 = ax.bar(r, positiveBars, color='#a1eba5ff', edgecolor='white', width=barWidth, label="Positif")
    # Create orange Bars
    p2 = ax.bar(r, netralBars, bottom=positiveBars, color='#f9ca95ff', edgecolor='white', width=barWidth, label="Netral")
    # Create blue Bars
    p3 = ax.bar(r, negatifBars, bottom=[i+j for i,j in zip(positiveBars, netralBars)], color='#ff8980ff', edgecolor='white', 
            width=barWidth, label="Negatif")

    # Custom x axis
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis = "y", color="#A8BAC4", lw=0.5)
    
    plt.xticks(r, names)
    plt.ylabel("Persentase")
    plt.bar_label(p1, label_type='center')
    plt.bar_label(p2, label_type='center')
    plt.bar_label(p3, label_type='center')
    plt.legend(loc='upper left', bbox_to_anchor=(0.95,1), ncol=1)
    
    
    stacked_temp = BytesIO()
    plt.savefig(stacked_temp, bbox_inches='tight', format='png')
    stacked_temp.seek(0)
    stacked_png = "data:image/png;base64,"
    stacked_png += base64.b64encode(stacked_temp.getvalue()).decode('utf-8')

    return stacked_png

def rating_bar(df_final):
    list_rating = {1 : [], 2 : [], 3 : [], 4 : [], 5 : []} # inisiasi dictionary

    for i in list_rating:    # membuat list banyak sentimen positif, netral, negatif
        list_rating[i].append(list(df_final[df_final['rating'] == i]['polaritas']).count('positive'))
        list_rating[i].append(list(df_final[df_final['rating'] == i]['polaritas']).count('netral'))
        list_rating[i].append(list(df_final[df_final['rating'] == i]['polaritas']).count('negative'))

    sentiment_tranpose = {                  # membuat sentimen sebagai kolomnya
            'positive' : [list_rating[i][0] for i in list_rating.keys()],
            'netral' : [list_rating[i][1] for i in list_rating.keys()],
            'negatif' : [list_rating[i][2] for i in list_rating.keys()],
            }

    idx = [1,2,3,4,5]
    df_sentiment_tranpose = pd.DataFrame(sentiment_tranpose)

    positiveBars = list(df_sentiment_tranpose['positive'])
    netralBars = list(df_sentiment_tranpose['netral'])
    negatifBars = list(df_sentiment_tranpose['negatif'])

    # plot
    barWidth = 0.8
    names = [i for i in list_rating.keys()]

    fig, ax = plt.subplots(figsize=(3,4))
    # Create green Bars
    p1 = ax.bar(idx, positiveBars, color='#a1eba5ff', edgecolor='white', width=barWidth, label="Positif")
    # Create orange Bars
    p2 = ax.bar(idx, netralBars, bottom=positiveBars, color='#f9ca95ff', edgecolor='white', width=barWidth, label="Netral")
    # Create blue Bars
    p3 = ax.bar(idx, negatifBars, bottom=[i+j for i,j in zip(positiveBars, netralBars)], color='#ff8980ff', edgecolor='white', 
            width=barWidth, label="Negatif")

    # Custom x axis
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis = "y", color="#A8BAC4", lw=0.5)

    plt.xticks(idx, names)
    plt.bar_label(p1, label_type='center')
    plt.bar_label(p2, label_type='center')
    plt.bar_label(p3, label_type='center')

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(0.75,1), ncol=1)
    rating_temp = BytesIO()
    plt.savefig(rating_temp, bbox_inches='tight', format='png')
    rating_temp.seek(0)
    rating_png = "data:image/png;base64,"
    rating_png += base64.b64encode(rating_temp.getvalue()).decode('utf-8')

    return rating_png

def pie_overall_sentiment(df_final):
    # create data
    names = ['Positif', 'Netral', 'Negatif']
    size = [list(df_final['polaritas']).count('positive'),
            list(df_final['polaritas']).count('netral'),
            list(df_final['polaritas']).count('negative')]

    # Create a circle at the center of the plot
    my_circle = plt.Circle( (0,0), 0.3, color='white')

    # Custom wedges
    plt.figure()
    # plt.title('Proporsi keseluruhan sentimen')
    plt.pie(size, labels=names, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, autopct='%1.1f%%',
            colors =['#a1eba5ff','#f9ca95ff','#ff8980ff'])

    p = plt.gcf()
    p.gca().add_artist(my_circle)
    # plt.show()
    pie_overall_temp = BytesIO()
    plt.savefig(pie_overall_temp, bbox_inches='tight',format='png')
    pie_overall_temp.seek(0)
    pie_overall_png = "data:image/png;base64,"
    pie_overall_png += base64.b64encode(pie_overall_temp.getvalue()).decode('utf-8')

    return pie_overall_png

def wordcloud_kata_sentimen(df):
    data  = dict(zip(df['kata'][:10].tolist(), df['jumlah'][:10].tolist()))

    wc = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure()
    # plt.title(f'15 kata teratas dalam kalimat {polaritas} aspek {aspek} ')
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    # plt.show()
    wordcloud_temp = BytesIO()
    plt.savefig(wordcloud_temp, bbox_inches='tight',format='png')
    wordcloud_temp.seek(0)
    wordcloud_png = "data:image/png;base64,"
    wordcloud_png += base64.b64encode(wordcloud_temp.getvalue()).decode('utf-8')

    return wordcloud_png

def jumlah_kemunculan_kata(list_kata):
    word_dict = {}
    for b in range(0,len(list_kata)):
        sentence = list_kata[b]
        word_token = word_tokenize(sentence)
        for j in word_token:
            if j not in word_dict:
                word_dict[j] = 1
            else:
                word_dict[j] += 1
    
    # membuat dataframe perhitungan kemunculan kata
    df = pd.DataFrame(word_dict.items(), columns=['kata', 'jumlah'])
    df.sort_values(by='jumlah', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    # membuat wordcloud kemunculan kata
    wc_viz = wordcloud_kata_sentimen(df)

    return wc_viz

def freq_analysis(df):
    list_kata = list(df)
    word_dict = {}
    for b in range(0,len(list_kata)):
        sentence = list_kata[b]
        word_token = word_tokenize(sentence)
        for j in word_token:
            if j not in word_dict:
                word_dict[j] = 1
            else:
                word_dict[j] += 1


    df_fr = pd.DataFrame(word_dict.items(), columns=['kata', 'jumlah'])
    df_fr.sort_values(by='jumlah', inplace=True, ascending=False)
    df_fr.reset_index(drop=True, inplace=True)
    df_fr.to_csv('static/df_fr.csv', index=False)