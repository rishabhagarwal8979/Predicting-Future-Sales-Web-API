import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

shops=pd.read_csv('shops_data.csv')
items=pd.read_csv('items_data.csv')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    features = [np.array(features)]
    temp=pd.DataFrame(features,columns=['shop_id','item_id','month','year'])
    if temp.loc[0,'shop_id']==0 :
        temp.loc[0,'shop_id']=57
    elif temp.loc[0,'shop_id']==1 :
        temp.loc[0,'shop_id']=58
    elif temp.loc[0,'shop_id']==10 :
        temp.loc[0,'shop_id']=11
    elif temp.loc[0,'shop_id']==40 :
        temp.loc[0,'shop_id']=39
        
    if temp.loc[0,'item_id'] in items['item_id']:
        temp['date_block_num']=(temp['year']-2013)*12+temp['month']-1
        days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
        temp["days"] = (temp["month"]-1).map(days).astype(np.int8)
        temp=pd.merge(temp,items,on=['item_id'],how='inner')
        temp=pd.merge(temp,shops,on=['shop_id'],how='inner')
        temp=temp[['date_block_num', 'shop_id', 'item_id', 'shop_category', 'shop_city', 'item_category_id', 'name2', 'name3', 'subtype_code', 'type_code', 'days', 'month', 'year', 'item_price']]
        prediction = abs(model.predict(temp))
    else :
        prediction = 0
        
    temp.to_csv('temp.csv')

    return render_template('index.html', prediction_text='Estimated number of sales will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)