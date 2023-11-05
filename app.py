import pickle
import streamlit as st

model = pickle.load(open('pizza-price-prediction.sav', 'rb'))


st.title('Prediksi Harga Pizza')

""" company = (0 = A, 1 = B, 2 = C, 3 = D, 4 = E)"""
Company = st.selectbox('Pilih Company: ', ['0','1','2','3','4'])

diameter = st.selectbox('Input diameter: ', ['22' , '20' , '16' , '14' , '18' , '18.5',  '8' , '12' , '16.5',  '8.5', '17'])

""" topping = (0 = chicken, 1 = papperoni, 2 = mushrooms, 3 = smoked beef, 4 = mozzarella, 5 = black papper, 6 = tuna, 
7 = meat, 8 = sausage, 9 = onion, 10 = vegetables, 11 = beef) """
topping = st.selectbox('Input Topping', ['0', '1', '2', '3', '4','5', '6', '7', '8', '9', '10','11'])

""" Variant = (0 = double_signature, 1 = american_favorite, 2 = super_supreme, 3 = meat_lovers, 4 = double_mix, 5 = classic, 6 = crunchy, 7 = new_york, 8 = double_decker, 9 = spicy_tuna, 10 = BBQ_meat_fiesta
, 11 = BBQ_sausage, 12 = extravaganza , 13 = meat_eater, 14 = gournet_greek, 15 = italian_veggie, 16 = thai_veggie, 17 = american_classic, 18 = neptune_tuna, 19 = spicy_tuna) """
variant = st.selectbox('Input Variant Pizza', ['0', '1', '2','3', '4', '5', '6', '7','8', '9', '10', '11','12', '13', '14', '15','16', '17', '18', '19'])

""" Size untuk pizza = ( 1 = jumbo, 2 = reguler, 3 = small, 4 = medium, 5 = large, 6 = XL) """
size = st.selectbox('Input Ukuran: ', ['1', '2', '3', '4', '5', '6'])

""" (1 = yes, 0 = no) """
extra_sauce = st.selectbox('Input tambahan saus:', ['1', '0'])

""" (1 = yes, 0 = no) """
extra_cheese = st.selectbox('Input tambahan keju: ', ['1', '0'])



predict = ''

if st.button('Estimasi Harga'):
    predict = model.predict(
        [[float(Company),float(diameter),float(topping),float(variant),float(size),float(extra_sauce),float(extra_cheese)]]
    )
    st.write('Prediksi Harga PIZZA : ', predict)
