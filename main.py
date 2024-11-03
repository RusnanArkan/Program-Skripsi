import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn import svm, metrics
import seaborn as sns
from preprocessing import preprocess_text, apply_tfidf, split_data, handle_imbalance
from sklearn.metrics import roc_curve, auc


# Initialize Streamlit container
with st.container():
    st.markdown(
        """
        <style>
        h1 {    
            text-align: center;
            font-family: 'Georgia', serif;
            white-space: nowrap;
            font-size: 45px;
            color: #FFFF;
            font-weight: normal;
        }
        
        
        </style>
        <h1>Analisis Sentimen Mobil Listrik</h1>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
    """
    <style>
    
    div.streamlit-expanderHeader {
        font-family: 'Georgia', serif;
        font-size: 18px;
        font-weight: normal;
        color: #FFFF;
    }

    </style>
    """,
    unsafe_allow_html=True
    )

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
    st.session_state['clf'] = None  # Model SVM
    st.session_state['tfidf'] = None  # TF-IDF vectorizer

# Bagian upload CSV
with st.expander('Upload CSV Disini!'):
    upl = st.file_uploader('Unggah file CSV atau Excel', type=['csv', 'xlsx'])

    if upl:
        st.write(f"File yang diunggah: {upl.name}")
        
        if upl.name.endswith('xlsx'):
            df = pd.read_excel(upl)
        elif upl.name.endswith('csv'):
            df = pd.read_csv(upl)

        st.write("Preview data:")
        st.write(df.head())

        # Memastikan kolom 'text_cleaning' dan 'sentimen' ada
        if 'text_cleaning' in df.columns and 'sentimen' in df.columns:
            
            # Preprocessing text
            df_filtered = df[df['sentimen'].isin(['positif', 'negatif'])]
            df_filtered['processed_text'] = df_filtered['text_cleaning'].apply(preprocess_text)

            # TF-IDF dan Data Splitting
            X, tfidf = apply_tfidf(df_filtered, 'processed_text')
            y = df_filtered['sentimen']
            X_train, X_test, y_train, y_test = split_data(X, y)

            # ADASYN imbalanced data
            X_train_resampled, y_train_resampled = handle_imbalance(X_train, y_train)
            
            # Perhitungan SVM
            clf = svm.SVC(max_iter=673, C=1, kernel='rbf', gamma=0.1)
            clf.fit(X_train_resampled, y_train_resampled)
            y_pred = clf.predict(X_test)
            
            # Menyimpan Model    
            st.session_state['clf'] = clf
            st.session_state['tfidf'] = tfidf
            st.session_state['model_trained'] = True

            st.success("Model berhasil dilatih menggunakan data CSV.")

            # Tombol untuk menampilkan akurasi dan hasil evaluasi
            if st.button('Tampilkan Akurasi'):
                
                st.write("Akurasi Test: ", clf.score(X_test, y_test)*100)
                

                # Classification report in dictionary format
                report = metrics.classification_report(y_test, y_pred, output_dict=True, digits=3)
                
                report_df = pd.DataFrame(report).transpose()

                # Menghilkangkan kolom 'support' jika muncul (optional)
                report_df = report_df.drop(columns=['support'], errors='ignore')

                # Grafik horizontal
                fig, ax = plt.subplots(figsize=(10, 6))
                report_df[['precision', 'recall', 'f1-score']].plot(kind='barh', ax=ax)
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f')
                
                plt.title('Classification Report Metrics')
                plt.xlabel('Score')
                plt.ylabel('Class')
                st.pyplot(fig)
                
                # Menampilkan matriks evaluasi
                st.write(report_df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
                    [{'selector': 'th', 'props': [('text-align', 'center')]}]
                    ))


            # Tombol untuk menampilkan visualisasi data
            if st.button('Tampilkan Data Visualisasi'):

                # Pie Chart untuk distribusi sentimen
                positif_count = df_filtered[df_filtered['sentimen'] == 'positif'].shape[0]
                negatif_count = df_filtered[df_filtered['sentimen'] == 'negatif'].shape[0]
                
                sentimen = [positif_count, negatif_count]
                labels = [f'Positif ({positif_count})', f'Negatif ({negatif_count})']
                colors = ['#55a868', '#c44e52']
                
                fig, ax = plt.subplots(figsize=(6, 3))
                plt.pie(sentimen, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title('Persentase Sentimen')
                plt.axis('equal')
                st.write("Pie Chart Persentase dari dataset berikut:")
                st.pyplot(plt)
                plt.clf()
 
                # Menampilkan hasil TF-IDF
                tfidf_features = tfidf.get_feature_names_out()  # Mendapatkan nama fitur (kata-kata)
                df_tfidf = pd.DataFrame(X.toarray(), columns=tfidf_features)  # Mengubah menjadi DataFrame

                st.write("Hasil TF-IDF (5 baris pertama):")
                st.write(df_tfidf.head())  # Menampilkan 5 baris pertama hasil TF-IDF

                # Setelah di Oversampling
                positif_count = (y_train_resampled == 'positif').sum()
                negatif_count = (y_train_resampled == 'negatif').sum()

                sentimen = [positif_count, negatif_count]
                labels = [f'Positif ({positif_count})', f'Negatif ({negatif_count})']
                colors = ['#55a868', '#c44e52']

                plt.pie(sentimen, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title('Persentase Sentimen (Setelah Oversampling)')
                plt.axis('equal')
                st.write("Pie Chart Setelah Dilakukan OverSampling Adasyn:")
                st.pyplot(plt)


                # WordCloud untuk positif dan negatif
                text_positive = ' '.join(df_filtered[df_filtered['sentimen'] == 'positif']['text_cleaning'])
                text_negative = ' '.join(df_filtered[df_filtered['sentimen'] == 'negatif']['text_cleaning'])

                wordcloud_positive = WordCloud(width=800, height=400).generate(text_positive)
                wordcloud_negative = WordCloud(width=800, height=400).generate(text_negative)

                plt.figure(figsize=(8, 4))
                plt.imshow(wordcloud_positive, interpolation='bilinear')
                plt.title('Wordcloud - Sentimen Positif')
                plt.axis('off')
                st.write("Wordcloud Sentimen positif:")
                st.pyplot(plt)

                plt.figure(figsize=(8, 4))
                plt.imshow(wordcloud_negative, interpolation='bilinear')
                plt.title('Wordcloud - Sentimen Negatif')
                plt.axis('off')
                st.write("Wordcloud Sentimen negatif:")
                st.pyplot(plt)

                # ROC curve
                
                # Prediksi probabilitas
                y_score = clf.decision_function(X_test)
                
                fpr, tpr, _ = roc_curve(y_test, y_score, pos_label='positif')
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc="lower right")
                st.pyplot(plt)

                # Confusion Matrix
                conf_matrix = metrics.confusion_matrix(y_test, y_pred)

                # Plot Confusion Matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix (Setelah Visualisasi)')
                st.pyplot(plt)

        else:
            st.error("File tidak memiliki kolom 'text_cleaning' atau 'sentimen'. Silakan upload file yang sesuai.")
            
# Bagian Input Teks
with st.expander("Input teks untuk diprediksi"):
    if not st.session_state['model_trained']:
        st.warning("Silakan unggah file CSV dan latih model terlebih dahulu.")
    else:
        user_input = st.text_area("Masukkan teks di sini:")

        if st.button('Prediksi Sentimen'):
            if user_input:
                # Preprocess input teks
                preprocessed_input = preprocess_text(user_input)
                
                # Transform input teks menggunakan TF-IDF yang sama
                X_input = st.session_state['tfidf'].transform([preprocessed_input])

                # Menggunakan model SVM yang telah dilatih untuk memprediksi sentimen
                y_pred_input = st.session_state['clf'].predict(X_input)

                # Menampilkan hasil prediksi
                if y_pred_input[0] == 'positif':
                    st.write("Hasil Prediksi Sentimen: **Positif**")
                else:
                    st.write("Hasil Prediksi Sentimen: **Negatif**")
            else:
                st.write("Silakan masukkan teks terlebih dahulu.")
