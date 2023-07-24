import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import flask
from flask import Flask, render_template, request, jsonify, url_for, session, redirect, flash
from flask_session import Session
import model as ml
import gensim.corpora as corpora
from statistics import mean
import json

app = Flask(__name__, template_folder='templates')
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/reset')
def reset():
	session.clear()
	return render_template('index.html')

@app.route('/r_train') 
def r_train():
	return render_template('training.html')

@app.route('/r_map', methods=['POST','GET'])
def r_map():
	return render_template('mapping.html')

@app.route('/r_db') 
def r_db():
	return render_template('dashboard.html')

@app.route('/training', methods=['POST','GET'])
def training():
	if request.method=="POST":
		session.clear()
		
		# input, read, process data
		file = request.files['file_train']
		df_raw = ml.read_data(file)
		df = ml.data_preprocessing(df_raw)

		# declare dictionary and corpus, train model
		id2word = corpora.Dictionary(df['t_kata'])
		corpus = [id2word.doc2bow(text) for text in df['t_kata']] # Create corpus Term Document Frequency (bag of words)
		model_lda, best_topics, df_lda_coherence, topics_range, selected_k, selected_a, selected_b = ml.optimasi_model(corpus,id2word, df)

		# visualisasi
		bt_viz = ml.best_topic_visualization(df_lda_coherence, topics_range) # visualisasi nilai coherence pada setiap jumlah topik
		list_wordcloud = ml.word_cloud_topic(model_lda)                      # list yg berisi wordcloud masing-masing kata pada setiap topik
		ml.pyldavis(model_lda, corpus)                                       # membuat visualisasi pyldavis 

		# declare session
		session["model_lda"] = model_lda
		session["df"] = df
		session["best_topics"] = best_topics        # untuk perulangan wordcloud topic
		session["list_wordcloud"] = list_wordcloud  
		session["selected_k"] = selected_k
		session["selected_a"] = selected_a
		session["selected_b"] = selected_b
		session["total_paragraf"] = df.shape[0]
		session["bt_viz"] = bt_viz
		session["table_lda"] = json.loads(df_lda_coherence.to_json(orient="split"))["data"]
		
		return jsonify({
	'best_topics':best_topics, 
	'bt_viz':bt_viz,
	'df_lda': json.loads(df_lda_coherence.to_json(orient="split"))["data"], 
	'column_df_lda':[{"title": str(col)} for col in json.loads(df_lda_coherence.to_json(orient="split"))["columns"]]
	})

@app.route('/map_viz', methods=['POST','GET'])
def map_viz():
	if request.method=="POST":

		# gets session
		model_lda = session.get('model_lda')
		df = session.get('df')

		# mendapatkan list aspek dari form mapping
		aspek = request.form.getlist('data[]') #list aspek yang dipilih berurutan sesuai dengan topik

		# map topic to aspect
		aspek_dict = ml.topic_aspect_mapping(model_lda, aspek)
		aspek_list = list(aspek_dict.keys())

		# map comment to aspect
		aspek = [i for i in aspek if i != '']
		review_aspect_dict = ml.comment_aspect_mapping(df, aspek_dict, aspek)

		# create dataframe
		df_tam = ml.topic_aspect_dataframe(df, review_aspect_dict, aspek)
		df_tam.to_csv('static/df_tam.csv', index=False)

		# declare lexicon
		lexicon = ml.lexicon
		lexicon = lexicon.drop(lexicon[(lexicon['word'] == 'bukan')
                               |(lexicon['word'] == 'tidak')
                               |(lexicon['word'] == 'ga')
                               |(lexicon['word'] == 'gak')
                               |(lexicon['word'] == 'gk')].index,axis=0)
		lexicon = lexicon[~lexicon.isin(['yang']).any(axis=1)]    # menghapus kata 'yang' karena bobot tidak sesuai
		lexicon = lexicon.reset_index(drop=True)
		lexicon_word = lexicon['word'].to_list()

		# final dataframe contain sentiment polarity
		df_final_raw = ml.sentiment_polarity_calculation(df_tam,lexicon_word)
		df_final_raw.to_csv('static/df_final_raw.csv', index=False)
		df_final_raw.insert(0,'no',[i for i in range(1,(df_final_raw.shape[0])+1)]) # menambahkan kolom nomor
		df_ulasan = df.explode('r_punctuation').reset_index(drop=True)
		df_final_raw.insert(1,'ulasan_tampil',df_ulasan['r_punctuation'])
		df_final = df_final_raw[['no','ulasan_tampil','ulasan','kategori','polaritas']] # df untuk tabel yang ditampilkan

		# final visualization
		bar_viz = ml.horizontal_bar_topic_per_aspect(aspek_dict,df_final_raw)
		stacked_viz = ml.stacked_bar_sentimen_per_aspect(df_final_raw, aspek)
		rating_viz = ml.rating_bar(df_final_raw)
		pie_viz = ml.pie_overall_sentiment(df_final_raw)

		aspek_list.insert(0, "umum")                    # list yang berisi semua kategori aspek ('umum', 'aspek1, 'aspek2', ..., dst)

		# df frequency for frequency analysis
		ml.freq_analysis(df_final_raw['ulasan'])

		# word cloud visualization
		polaritas = ['positive', 'negative']
		wc_list = []                                    # list untuk menyimpan grafik wordcloud
		for i in range(len(aspek_list)):
			wc_list_tmp = []
			for j in polaritas:
				if aspek_list[i] == 'umum':
					wc_viz = ml.jumlah_kemunculan_kata(list(df_final_raw[df_final_raw['polaritas'] == j]['ulasan']))
					wc_list_tmp.append(wc_viz)
				else:
					wc_viz = ml.jumlah_kemunculan_kata(list(df_final_raw[(df_final_raw[aspek_list[i]] == 1) & (df_final_raw['polaritas'] == j)]['ulasan']))
					wc_list_tmp.append(wc_viz)
			wc_list.append(wc_list_tmp)

		
		# variable yang nantinya akan dilakukan perulangan
		total_ulasan = []
		rating_ulasan = []
		ulasan_positif = []
		ulasan_negatif = []
		ulasan_netral = []
		pie_viz_list = []
		rating_viz_list = []

		for i in range(len(aspek_list)):
			if aspek_list[i] == 'umum':
				total_ulasan.append(df_final_raw.shape[0])
				rating_ulasan.append(round(mean(df_final_raw['rating']),2))
				ulasan_positif.append(df_final_raw[df_final_raw['polaritas'] == 'positive'].shape[0])
				ulasan_negatif.append(df_final_raw[df_final_raw['polaritas'] == 'negative'].shape[0])
				ulasan_netral.append(df_final_raw[df_final_raw['polaritas'] == 'netral'].shape[0])
				pie_viz_list.append(ml.pie_overall_sentiment(df_final_raw))
				rating_viz_list.append(ml.rating_bar(df_final_raw))
			else:
				total_ulasan.append(df_final_raw[df_final_raw[aspek_list[i]]==1].shape[0])
				rating_ulasan.append(round(mean(df_final_raw[df_final_raw[aspek_list[i]]==1]['rating']),2))
				ulasan_positif.append(df_final_raw[(df_final_raw[aspek_list[i]] == 1) & (df_final_raw['polaritas'] == 'positive')].shape[0])
				ulasan_negatif.append(df_final_raw[(df_final_raw[aspek_list[i]] == 1) & (df_final_raw['polaritas'] == 'negative')].shape[0])
				ulasan_netral.append(df_final_raw[(df_final_raw[aspek_list[i]] == 1) & (df_final_raw['polaritas'] == 'netral')].shape[0])
				pie_viz_list.append(ml.pie_overall_sentiment(df_final_raw[df_final_raw[aspek_list[i]] == 1]))
				rating_viz_list.append(ml.rating_bar(df_final_raw[df_final_raw[aspek_list[i]] == 1]))

		# declare session
		session["total_ulasan"] = total_ulasan
		session["rating_ulasan"] = rating_ulasan
		session["ulasan_positif"] = ulasan_positif
		session["ulasan_negatif"] = ulasan_negatif
		session["ulasan_netral"] = ulasan_netral
		session["pie_viz_list"] = pie_viz_list
		session["rating_viz_list"] = rating_viz_list

		# declare session visualization
		session["aspek_list"] = aspek_list
		session["bar_viz"] = bar_viz
		session["stacked_viz"] = stacked_viz
		session["pie_viz"] = pie_viz
		session["rating_viz"] = rating_viz
		session["wc_viz_list"] = wc_list

		# declare session table
		table_session = []
		for i in range(len(aspek_list)):
			if aspek_list[i] == 'umum':
				session["table_umum"] = json.loads(df_final[['no','ulasan_tampil','kategori','polaritas']].to_json(orient="split"))["data"]
				table_session.append(f"{session['table_umum']}")
			else:
				session[f"table_{aspek_list[i]}"] = json.loads(df_final_raw[df_final_raw[aspek_list[i]]==1][['no','ulasan_tampil','polaritas']].to_json(orient="split"))["data"]
				table_session.append(f"{session[f'table_{aspek_list[i]}']}")

		session["table_session"] = table_session

		# fungsi harus mengembalikan sesuatu untuk dapat berjalan
		# sehingga dikembalikan sebuah data dalam bentuk json, akan tetapi nantinya tidak akan dipakai
		# data yang dipakai nanti adalah data dari sesssion
		return jsonify({
	'aspek':aspek, 
	'aspek_dict':aspek_dict,
	'bar_viz':bar_viz,
	'stacked_viz':stacked_viz,
	'pie_viz':pie_viz,
	'rating_viz':rating_viz,
	'df_final': json.loads(df_final.to_json(orient="split"))["data"], 
	'column_df_final':[{"title": str(col)} for col in json.loads(df_final.to_json(orient="split"))["columns"]]})

@app.route('/pyldavis', methods=['POST','GET'])
def lda_viz():
	return flask.send_file('static/lda.html')

if __name__ == "__main__":
	app.run(debug=True)
