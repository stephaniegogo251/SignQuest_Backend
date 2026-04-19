import sys
import os
import json
import numpy as np
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'asl_model.h5')
ESL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'esl_model.h5')

# load asl and esl model
MODELS = {
    'ASL': load_model(ASL_MODEL_PATH),
    'ESL': load_model(ESL_MODEL_PATH)
}

print("Ready")

# labels for the sign list
SIGNS = None

asl_signs = ['1', '3', 'bye', 'can', 'coffee', 'doctor', 'fainted', 'he', 
            'hello', 'help', 'hot', 'hotel', 'i', 'idle', 'lost', 'needs', 
            'nice', 'no', 'please', 'small', 'thank you', 'this', 'ticket', 
            'to meet', 'too', 'travelling', 'wait', 'walk', 'want', 'with', 
            'yes', 'you', 'your welcome']

esl_signs = ['1','50','card','emirates id','excellent','fine','finish','how are you','i','idle',
            'know','laqeemat','majboos','near','no','plate','room','salaam alaikum','small',
            'sorry','thank you','travel','want','yes']


def main():
    global SIGNS

    while True:
        line = sys.stdin.readline() # read data sent from Node.js
        if not line:
            break
        
        try:
            data = json.loads(line)
            path = data.get('path', 'ASL').upper()  # Default to ASL if not provided
            if path == 'ASL':
                SIGNS = np.array(asl_signs)
            else:
                SIGNS = np.array(esl_signs)
            keypoints = np.array(data['keypoints']) # Shape (30, 258)

            # Select the correct model based on user's current level path
            model = MODELS.get(path)
            
            # predicts user's keypoints
            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            
            # Get the predicted sign's confidence score
            predicted_sign = SIGNS[np.argmax(res)]
            confidence = float(np.max(res))

            # sends json back to Node.js
            output = {
                "sign": predicted_sign,
                "confidence": confidence
            }
            sys.stdout.write(json.dumps(output) + '\n')
            sys.stdout.flush() #sends data immeduately

        except Exception as e:
            error_output = {"error": str(e)}
            sys.stdout(json.dumps(error_output) + '\n')
            sys.stdout.flush()

if __name__ == "__main__":
    main()