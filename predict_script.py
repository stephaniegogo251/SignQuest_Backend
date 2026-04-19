import sys
import json
import numpy as np
from keras.models import load_model

asl_signs = ['1', '3', 'bye', 'can', 'coffee', 'doctor', 'fainted', 'he', 
            'hello', 'help', 'hot', 'hotel', 'i', 'idle', 'lost', 'needs', 
            'nice', 'no', 'please', 'small', 'thank you', 'this', 'ticket', 
            'to meet', 'too', 'travelling', 'wait', 'walk', 'want', 'with', 
            'yes', 'you', 'your welcome']

esl_signs = ['1','50','card','emirates id','excellent','fine','finish','how are you','i','idle',
            'know','laqeemat','majboos','near','no','plate','room','salaam alaikum','small',
            'sorry','thank you','travel','want','yes']


asl_path = "https://signquest-backend-yhj4.onrender.com/static/models/aslmodel.h5"
esl_path = "https://signquest-backend-yhj4.onrender.com/static/models/eslmodel.h5"

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        try:
            data = json.loads(line)
            path = data.get('path', 'ASL').upper()
            keypoints = np.array(data['keypoints'])

            model = load_model(asl_path) if path == 'ASL' else load_model(esl_path)
            signs_list = asl_signs if path == 'ASL' else esl_signs
            
            # Prediction
            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            
            predicted_sign = signs_list[np.argmax(res)]
            confidence = float(np.max(res))

            output = {
                "sign": str(predicted_sign),
                "confidence": confidence
            }
            sys.stdout.write(json.dumps(output) + '\n')
            sys.stdout.flush()

        except Exception as e:
            # Crucial: Use .write and ensure it's JSON
            sys.stdout.write(json.dumps({"error": str(e)}) + '\n')
            sys.stdout.flush()

if __name__ == "__main__":
    main()