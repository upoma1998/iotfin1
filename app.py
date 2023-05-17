import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("IOT.csv")
data = np.array(data)

X = data[1:, 0:-1]
y = data[1:, -1]
y = y.astype('str')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)
pickle.dump(log_reg,open('model.pkl','wb'))

import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))


def predict_forest(Nitrogen,Phosphorus,Potassium,SoilTemperature,Moisture,pH,Humidity,AirTemperature):
    input=np.array([[Nitrogen,Phosphorus,Potassium,SoilTemperature,Moisture,pH,Humidity,AirTemperature]]).astype(np.float64)
   
    prediction = log_reg.predict(input)
    print(prediction)
    return prediction

def apply_formulaN(x):
    x=float(x)
    if x >=12 and x<=38.75:
      p = (180-((45/26.75)*(x-12)))
      return abs(p*2.174)
    elif x >38.75 and x <= 65.5:
      p = (135-((45/26.75)*(x-38.75)))
      return abs(p*2.174)
    elif x >65.5 and x <=92.25:
      p = (90-((45/26.75)*(x-65.5)))
      return abs(p*2.174)
    elif x >92.25 and x <=119:
      p = (45-((45/26.75)*(x-92.25)))
      return abs(p*2.174)
    else:
      return 0
def apply_formulaP(y):
    y=float(y)
    if y>=5 and y<=33.75:
      p1 = (40-((10/28.75)*(y-5)))
      return abs(p1*5)
    elif y >33.75 and y <= 62.5:
      p1 = (30-((10/28.75)*(y-33.75)))
      return abs(p1*5)
    elif y >62.5 and y <=91.25:
      p1 = (20-((10/28.75)*(y-62.5)))
      return abs(p1*5)
    elif y >91.25 and y <=120:
      p1 = (10-((10/28.75)*(y-91.25)))
      return abs(p1*5)
    else:
      return 0
def apply_formulaK(z):
    z=float(z)
    if z>=25 and z<=43.75:
      p2 = (180-((45/18.75)*(z-25)))
      return abs(p2*2)
    elif z >43.75 and z <= 62.5:
      p2 = (135-((45/18.75)*(z-43.75)))
      return abs(p2*2)
    elif z >62.5 and z <=81.25:
      p2 = (90-((45/18.75)*(z-62.5)))
      return abs(p2*2)
    elif z >81.25 and z <=100:
      p2 = (45-((45/18.75)*(z-81.25)))
      return abs(p2*2)
    else:
      return 0
    


    

def main():
    st.title("IoT based Real Time Soil Sensing for Precision Agriculture")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Crop Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Nitrogen = st.text_input("Nitrogen","Type Here")
    Phosphorus = st.text_input("Phosphorus","Type Here")
    Potassium = st.text_input("Potassium","Type Here")
    SoilTemperature = st.text_input("Soil Temperature","Type Here")
    Moisture = st.text_input("Moisture","Type Here")
    pH = st.text_input("pH","Type Here")
    Humidity = st.text_input("Humidity","Type Here")
    AirTemperature = st.text_input("Air Temperature","Type Here")
    
   

    if st.button("Predict"):
        output=predict_forest(Nitrogen,Phosphorus,Potassium,SoilTemperature,Moisture,pH,Humidity,AirTemperature)
        print(output)
        urea=apply_formulaN(Nitrogen)
        tsp=apply_formulaP(Phosphorus)
        mop=apply_formulaK(Potassium)
        #print(list(output)[0])
        #lbl =list(output)[0]
        #lbl = (list(output)[0])
        #coffee_bags_as_int = int(float(coffee_bags))
        #x = np.int8(list(output)[0])
        st.markdown("**###The crop having maximum production will be {}**".format(output))
        st.markdown("**###For the maximum production of {} the amount of Urea in kg/ha is {}**".format(output,urea))
        
        st.markdown("**###For the maximum production of {} the amount of TSP in kg/ha is {}**".format(output,tsp))
        st.markdown("**###For the maximum production of {} the amount of MoP in kg/ha is {}**".format(output,mop))


if __name__=='__main__':
    main()