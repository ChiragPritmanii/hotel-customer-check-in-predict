from flask import Flask, render_template, request
import pycountry_convert as pc
from tensorflow.keras.models import load_model
import pickle
import os
import numpy as np


app = Flask(__name__)
model = load_model('ann_model.hdf5', compile=False)

enc_agegroup = np.load('enc_agegroup.npy', allow_pickle=True)
enc_continent = np.load('enc_continent.npy', allow_pickle=True)
enc_disb= np.load('enc_disb.npy', allow_pickle=True)
enc_mkt = np.load('enc_mkt.npy', allow_pickle=True)
enc_nationality = np.load('enc_nationality.npy', allow_pickle=True)
enc_recordcreation = np.load('enc_recordcreation.npy', allow_pickle=True)
l2_norms_dict = pickle.load(open('l2_norms_dict.sav', 'rb')) 


@app.route('/', methods=['GET'])
def Home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        check_overseas = lambda x: 0 if x=='EU' else 1
        check_countryman = lambda x: 1 if x=='PRT' else 1
        check_adult = lambda x: 1 if x>=18 else 0
        check_NearElevator = lambda x: 1 if x=='near' else 0
        check_AwayElevator = lambda x: 1 if x=='away' else 0
        convert_to_int = lambda x: 1 if x=='yes' else 0
        def check_AgeGroup(age):
            age_groups = {'child':range(1,13), 'teen':range(13,18), 'young_adult': range(18,30), 'early_middle_aged': range(30,45), 
                  'late_middle_aged':range(45,65), 'old': range(65,100)}
            for k,v in age_groups.items():
                if age in v:
                    return k

        def partition_DaysSinceCreation(x):
            if x<=365:
                val = 'within_year'
            elif x in range(366,731):
                val = 'more_than_year_ago' 
            else:
                val = 'more_than_3years_ago'
            return val
        
        def get_continent_name(country_code):
            if country_code == 'TMP':
                return 'AS'
            elif country_code == 'ATA' or country_code == 'ATF':
                return 'AN'
            elif country_code == 'UMI':
                return 'NA'
            elif country_code == 'PCN':
                return 'OC'
            else:
                alpha2 = pc.country_alpha3_to_country_alpha2(country_code)
                continent_name = pc.country_alpha2_to_continent_code(alpha2)
                return continent_name

        NATIONALITY = str(request.form['nationality'])
    
        CONTINENT = get_continent_name(NATIONALITY)
        
        OVERSEAS = check_overseas(CONTINENT)
        
        COUNTRYMAN = check_countryman(NATIONALITY)
        
        AGE = int(request.form['age'])
        
        ADULT = check_adult(AGE)
        
        AGEGROUP = check_AgeGroup(AGE)
        
        DAYSSINCECREATION = int(request.form['days since creation'])
        RECORDCREATEDSINCE = partition_DaysSinceCreation(DAYSSINCECREATION)
        BOOKINGSCANCELED = int(request.form['bookings canceled'])
        BOOKINGSNOSHOWED = int(request.form['bookings no showed'])
        BOOKINGSNOTCHECKEDIN = sum([BOOKINGSCANCELED, BOOKINGSNOSHOWED])
        DISTRIBUTIONCHANNEL = str(request.form['distribution'])
        MARKETSEGMENT = str(request.form['market seg'])
        HIGHFLOOR = convert_to_int(str(request.form['high floor']))
        LOWFLOOR = convert_to_int(str(request.form['low floor']))
        MEDIUMFLOOR = convert_to_int(str(request.form['medium floor']))
        ACCESSIBLEROOM = convert_to_int(str(request.form['accessible room']))
        BATHTUB = convert_to_int(str(request.form['bathtub']))
        SHOWER = convert_to_int(str(request.form['shower']))
        CRIB = convert_to_int(str(request.form['crib']))
        KINGSIZEBED = convert_to_int(str(request.form['kingsize bed']))
        TWINBED = convert_to_int(str(request.form['twin bed']))
        ELEVATORDIST = str(request.form['elevator distance'])
        NEARELEVATOR = check_NearElevator(ELEVATORDIST)
        AWAYFROMELEVATOR = check_AwayElevator(ELEVATORDIST)
        NOALCOHOLINMINIBAR = convert_to_int(str(request.form['no alcohol']))
        QUIETROOM = convert_to_int(str(request.form['quiet room']))
        TOTALSR = sum([HIGHFLOOR,LOWFLOOR,ACCESSIBLEROOM,MEDIUMFLOOR,BATHTUB, SHOWER, CRIB, 
                        KINGSIZEBED, TWINBED, NEARELEVATOR, AWAYFROMELEVATOR, NOALCOHOLINMINIBAR, QUIETROOM])

        print(NATIONALITY,DISTRIBUTIONCHANNEL,MARKETSEGMENT,CONTINENT,AGEGROUP,RECORDCREATEDSINCE,HIGHFLOOR,LOWFLOOR,ACCESSIBLEROOM,
        MEDIUMFLOOR,BATHTUB, SHOWER, CRIB, KINGSIZEBED, TWINBED, NEARELEVATOR, AWAYFROMELEVATOR, NOALCOHOLINMINIBAR, QUIETROOM,OVERSEAS, 
        COUNTRYMAN, ADULT, AGE,DAYSSINCECREATION,BOOKINGSCANCELED,BOOKINGSNOSHOWED, TOTALSR, BOOKINGSNOTCHECKEDIN)

        NATIONALITY =  np.where(enc_nationality==NATIONALITY)[0][0]
        AGEGROUP = np.where(enc_agegroup==AGEGROUP)[0][0]
        CONTINENT = np.where(enc_continent==CONTINENT)[0][0]
        DISTRIBUTIONCHANNEL = np.where(enc_disb==DISTRIBUTIONCHANNEL)[0][0]
        MARKETSEGMENT = np.where(enc_mkt==MARKETSEGMENT)[0][0]
        RECORDCREATEDSINCE = np.where(enc_recordcreation==RECORDCREATEDSINCE)[0][0]
        AGE /= l2_norms_dict['Age']
        DAYSSINCECREATION /= l2_norms_dict['DaysSinceCreation']
        BOOKINGSCANCELED /= l2_norms_dict['BookingsCanceled']
        BOOKINGSNOSHOWED /= l2_norms_dict['BookingsNoShowed']
        TOTALSR /= l2_norms_dict['TotalSR']


        print(NATIONALITY,DISTRIBUTIONCHANNEL,MARKETSEGMENT,CONTINENT,AGEGROUP,RECORDCREATEDSINCE,HIGHFLOOR,LOWFLOOR,ACCESSIBLEROOM,
        MEDIUMFLOOR,BATHTUB, SHOWER, CRIB, KINGSIZEBED, TWINBED, NEARELEVATOR, AWAYFROMELEVATOR, NOALCOHOLINMINIBAR, QUIETROOM,OVERSEAS, 
        COUNTRYMAN, ADULT, AGE,DAYSSINCECREATION,BOOKINGSCANCELED,BOOKINGSNOSHOWED, TOTALSR, BOOKINGSNOTCHECKEDIN)

        query = np.array([[NATIONALITY,DISTRIBUTIONCHANNEL,MARKETSEGMENT,CONTINENT,AGEGROUP,RECORDCREATEDSINCE,HIGHFLOOR,LOWFLOOR,ACCESSIBLEROOM,
        MEDIUMFLOOR,BATHTUB, SHOWER, CRIB, KINGSIZEBED, TWINBED, NEARELEVATOR, AWAYFROMELEVATOR, NOALCOHOLINMINIBAR, QUIETROOM,OVERSEAS, 
        COUNTRYMAN, ADULT, AGE, DAYSSINCECREATION,BOOKINGSCANCELED,BOOKINGSNOSHOWED, TOTALSR, BOOKINGSNOTCHECKEDIN]])

        print(query)
        prediction = model.predict([query[:, 0].reshape(-1,1), query[:,1].reshape(-1,1), query[:,2].reshape(-1,1), query[:,3].reshape(-1,1), query[:,4].reshape(-1,1), query[:,5].reshape(-1,1), query[:,6:]])
        print(prediction)
        output = round(prediction[0][0]*100,2)
        print(output)
        return render_template('home.html', prediction=f"There is {output}% chance that the customer would check-in")

    else:
        return render_template('home.html')


if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=8080)