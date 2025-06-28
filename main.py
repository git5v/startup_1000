import streamlit as st
import numpy as np # for performing mathematical calculations behind ML algorithms
import matplotlib.pyplot as plt # for visualization
import pandas as pd # for handling and cleaning the dataset
import seaborn as sns # for visualization
import sklearn # for model evaluation and development

string = "Startup's Profit Prediction"

st.set_page_config(page_title=string, page_icon="✅", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title (string, anchor=None)
st.write("""

            - By *Vedant Mukhedkar* :sunglasses: 

""")


from PIL import Image
image = Image.open('startup.png')

st.image(image)


df = pd.read_csv("1000_Companies.csv")

# # spliting Dataset in Dependent & Independent Variables
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 4].values

def drop_outliers(train,columns):
    iqr=1.5+(np.percentile(train[columns],75)-np.percentile(train[columns],25))
    df.drop(train[train[columns]>(iqr+np.percentile(train[columns],75))].index,inplace=True)
    df.drop(train[train[columns]<(np.percentile(train[columns],25)-iqr)].index,inplace=True)
drop_outliers(df,"Profit")
drop_outliers(df,"Administration")

df.replace({'State':{'California':0,'New York':1,'Florida':2}},inplace=True)
df.head()

X=df.drop(['Profit'],axis=1)
Y=df['Profit']

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=2)

# Create the regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit to training data
rf_regressor.fit(X_train, Y_train)


y_pred_rf = rf_regressor.predict(X_test)


# Mean Squared Error
mse = mean_squared_error(Y_test, y_pred_rf)
# R^2 Score
r2 = r2_score(Y_test, y_pred_rf)

print(f"Random Forest Regression MSE: {mse:.2f}")
print(f"Random Forest Regression R^2: {r2:.2f}")




# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])


# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(x_train,y_train)

# y_pred = model.predict(x_test)

# testing_data_model_score = model.score(x_test, y_test)
# print("Model Score/Performance on Testing data",testing_data_model_score)

# training_data_model_score = model.score(x_train, y_train)
# print("Model Score/Performance on Training data",training_data_model_score)

rnd_cost = st.sidebar.number_input('Insert R&D Spend')
st.write('The current number is ', rnd_cost)

Administration_cost = st.sidebar.number_input('Insert Administration cost Spend')
st.write('The current number is ', Administration_cost)

Marketing_cost_Spend = st.sidebar.number_input('Insert Marketing cost Spend')
st.write('The current number is ', Marketing_cost_Spend)

option = st.sidebar.selectbox(
     'Select the region',
     ('Delhi', 'Banglore', 'Pune'))

st.write('You selected:', option)

if option == "Pune":
    optn = 0
if option == "Banglore":
    optn = 1
if option == "Delhi":
    optn = 2   

total_value = rnd_cost+Administration_cost+Marketing_cost_Spend
y_pred = rf_regressor.predict([[Marketing_cost_Spend,Administration_cost,rnd_cost,optn]])

if st.button('Predict'):
    st.success('The Profit must be  {} '.format(y_pred))
else:
     st.write('Please fill all the important details')


fig = plt.figure()

X = ['Toal cost Spend']
x_value = [rnd_cost+Administration_cost+Marketing_cost_Spend]
  
X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, x_value, 0.4, label = 'cost')
# plt.bar(X_axis + 0.2, y_pred, 0.4, label = 'profit')
  
# plt.xticks(X_axis, X)
# plt.xlabel("RS")
# plt.title("Profit vs Toal cost spend")
# plt.legend()
# plt.show()

# st.pyplot(fig)



fig, ax = plt.subplots()
bars1 = ax.bar(X_axis - 0.2, x_value, 0.4, label='cost')
bars2 = ax.bar(X_axis + 0.2, y_pred, 0.4, label='profit')

# Add value labels to each bar
ax.bar_label(bars1, padding=3, fmt='%.2f')  # for cost bars
ax.bar_label(bars2, padding=3, fmt='%.2f')  # for profit bars

ax.set_xlabel('Dollars')
ax.set_ylabel('Value')
ax.set_title('Profit vs Toal cost spend')
ax.legend()
plt.show()
st.pyplot(fig)


plt.bar(X_axis, x_value, width=0.4, label='cost')
plt.bar(X_axis, y_pred, width=0.4, bottom=x_value, label='profit')
plt.xlabel('Dollars')
plt.ylabel('Value')
plt.legend()
plt.title('Stacked Bar Chart: Cost + Profit')
plt.show()

st.pyplot(fig)

if total_value!=0:
    percent_change = ((y_pred - total_value) / total_value) * 100
    st.write(f"percentage in change {float(percent_change):.2f}%")
