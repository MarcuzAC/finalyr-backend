from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model_filename.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    
    # Extract features from JSON data
    try:
        gender = data['gender']
        parentaledu = data['parentaledu']
        lunch = data['lunch']
        prepcourse = data['prepcourse']
        mathscore = data['mathscore']
        physicsscore = data['physicsscore']
        chemscore = data['chemscore']
        studytime = data['studytime']
        walkdist = data['walkdist']
        absences = data['absences']
        gpa = data['gpa']
        # grade = data['grade']
        
        # Convert categorical features to numerical values
        # For simplicity, replace these mappings with your actual preprocessing logic
        # gender = 0 if gender == 'female' else 1
        # parentaledu_mapping = {
        #     "bachelor's degree": 0,
        #     'some college': 1,
        #     "master's degree": 2,
        #     "associate's degree": 3,
        #     'high school': 4,
        #     'some high school': 5
        # }
        # parentaledu = parentaledu_mapping.get(parentaledu, -1)
        # lunch = 0 if lunch == 'standard' else 1
        # prepcourse = 0 if prepcourse == 'none' else 1
        
        # Prepare the feature array
        features = [gender, parentaledu, lunch, prepcourse, mathscore, physicsscore, chemscore, studytime, walkdist, absences, gpa]
        
        # Make prediction
        prediction = model.predict([features])
        
        return jsonify({'prediction': prediction[0]})
    
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

