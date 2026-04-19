import sys
import json
import numpy as np
from keras.models import load_model

# load asl and esl model
MODELS = {
    'ASL': load_model('models/asl_model.h5'),
    'ESL': load_model('models/esl_model.h5')
}

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
            path = data.get('path', 'ASL')  # Default to ASL if not provided
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
            print(json.dumps(output))
            sys.stdout.flush() #sends data immeduately

        except Exception as e:
            error_output = {"error": str(e)}
            print(json.dumps(error_output))
            sys.stdout.flush()

if __name__ == "__main__":
    main()