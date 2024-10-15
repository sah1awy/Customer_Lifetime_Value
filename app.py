import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image


rf = pickle.load(open('rf.pkl','rb'))
ann = load_model('ann.h5')
sc = pickle.load(open('scaler1.pkl','rb'))
qn = pickle.load(open('scaler2.pkl','rb'))


def predict_clv(freq,ex,ex_avg,rec,t):
    x1 = sc.transform([[freq,ex,ex_avg]])
    x2 = qn.transform([[rec,t]])
    x_final = np.zeros((1, 5), dtype=np.float32)
    x_final[0, 0] = x1[0, 0]  # First position from x1
    x_final[0, 1] = x2[0, 0]  # Second position from x2 (rec)
    x_final[0, 2] = x2[0, 1]  # Third position from x2 (t)
    x_final[0, 3] = x1[0, 1]  # Fourth position from x1 (ex)
    x_final[0, 4] = x1[0, 2]  # Fifth position from x1 (ex_avg)

    pred1 = rf.predict(x_final)
    pred2 = ann.predict(x_final)
    pred = (pred1[0] + pred2[0,0]) / 2
    return pred


def main():
    st.title("Customer Lifetime Value Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Customer Lifetime Value Predictor </h2>
    </div>
    """
    # ['frequency', 'expected_purchases', 'expected_avg_value'],['recency', 'T']
    st.markdown(html_temp,unsafe_allow_html=True)
    freq = st.text_input("Frequency","Type")
    ex = st.text_input("Expected Purchases","Type")
    ex_avg = st.text_input("Expected Avg Value","Type")
    rec = st.text_input("Recency","Type")
    t = st.text_input("T","Type")
    res = 0
    if st.button("Predict"):
        res = predict_clv(freq,ex,ex_avg,rec,t)
    st.success('The output is {:.2f}'.format(res))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Build with Streamlit")


if __name__ == "__main__":
    main()

