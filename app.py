import pickle
import streamlit as st

model = pickle.load(open('pizza-price-prediction.sav', 'rb'))


st.title('Prediksi Harga Pizza')


Company = st.selectbox('Pilih Company: ', ['0','1','2','3','4'])


diameter = st.selectbox('Input diameter: ', ['22' , '20' , '16' , '14' , '18' , '18.5',  '8' , '12' , '16.5',  '8.5', '17'])


topping = st.selectbox('Input Topping', ['0', '1', '2', '3', '4','5', '6', '7', '8', '9', '10','11'])


variant = st.selectbox('Input Variant Pizza', ['0', '1', '2','3', '4', '5', '6', '7','8', '9', '10', '11','12', '13', '14', '15','16', '17', '18', '19'])


size = st.selectbox('Input Ukuran: ', ['1', '2', '3', '4', '5', '6'])

extra_sauce = st.selectbox('Input tambahan saus:', ['1', '0'])

extra_cheese = st.selectbox('Input tambahan keju: ', ['1', '0'])



predict = ''

if st.button('Estimasi Harga'):
    predict = model.predict(
        [[float(Company),float(diameter),float(topping),float(variant),float(size),float(extra_sauce),float(extra_cheese)]]
    )
    st.write('Prediksi Harga PIZZA : ', predict)