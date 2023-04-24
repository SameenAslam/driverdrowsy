import flask
from pyngrok import ngrok

from flask import request, jsonify
import io, base64
from PIL import Image
# from detector import detect_drowsiness
from detector import detect_drowsiness 
import os
from datetime import datetime


# detect_drowsiness("asd")

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def check_authorization(request):
    key = "@H#U5UzrCO@n27QlmGJPSAi&iK1"
    headers = request.headers

    auth = headers.get("x-api-fyp-key")

    if auth == key:
        return True
    else:
        return False


@app.route('/api/v1/predict', methods=['POST'],)
def home():
    if check_authorization(request):
        data = request.json
      #   data
      #   print()

      #   return jsonify({"result" : True})
      #   with open('../requests/test3.txt', 'w') as f:
      #       f.write(str(data["image"]))
       # Assuming base64_str is the string value without 'data:image/jpeg;base64,'
        dt = datetime.now()
        time_stamp = str(datetime.timestamp(dt))
        image_file_name = './my_image_'+data['device_id']+'_'+time_stamp.replace("-","_").replace(".","_")+'.jpg'
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(data['image'], "utf-8"))))
        rgb_im = img.convert('RGB')
        img.save(image_file_name)
        res = detect_drowsiness(image_file_name)

        os.unlink(image_file_name)
        if res['error'] != "none":
            return jsonify(res), 400
        return jsonify(res), 200
      #   response =  {
      #    "FaceDetails": [
      #       {
      #          "AgeRange": {
      #             "High": 22,
      #             "Low": 18
      #          },
      #          "Beard": {
      #             "Confidence": 0.8,
      #             "Value": True
      #          },
      #          "BoundingBox": {
      #             "Height": 100,
      #             "Left": 21,
      #             "Top": 32,
      #             "Width": 123
      #          },
      #          "Confidence": 99,
      #          "Emotions": [
      #             {
      #                "Confidence": 99,
      #                "Type": "happy"
      #             }
      #          ],
      #          "Eyeglasses": {
      #             "Confidence": 10,
      #             "Value": False
      #          },
      #          "EyesOpen": {
      #             "Confidence": 90,
      #             "Value": True
      #          },
      #          "Gender": {
      #             "Confidence": 100,
      #             "Value": "unknown"
      #          },
      #          "Landmarks": [
      #             {
      #                "Type": "dsf",
      #                "X": 123,
      #                "Y": 432
      #             }
      #          ],
      #          "MouthOpen": {
      #             "Confidence": 1,
      #             "Value": False
      #          },
      #          "Mustache": {
      #             "Confidence": 100,
      #             "Value": True
      #          },
      #          "Pose": {
      #             "Pitch": 321,
      #             "Roll": 123,
      #             "Yaw": 32
      #          },
      #          "Quality": {
      #             "Brightness": 123,
      #             "Sharpness": 123
      #          },
      #          "Smile": {
      #             "Confidence": 100,
      #             "Value": False
      #          },
      #          "Sunglasses": {
      #             "Confidence": 100,
      #             "Value": True
      #          }
      #       }
      #    ],
      #    "OrientationCorrection": "string"
      # }
      #   return jsonify(response), 200
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401
    
    # return {"data":data,"authorization_header":authorization_header}



app.run(port=8000)