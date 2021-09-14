from datetime import datetime
from flask import Flask, request, render_template
from inference import get_category,save_image

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def fragment(): 
    # Write the GET Method to get the index file
    if request.method == 'GET':
        return render_template('index.html')
    # Write the POST Method to post the results file
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('File Not Uploaded')
            return
        # Read file from upload
        file = request.files['file']
        save_image(file,"input")
        # Get category of prediction
        model1 = 'modelDeepLabV3_Mila.tflite'
        model2 = 'lite-model_deeplabv3-xception65_1_default_2.tflite'
        model3 = 'lite-model_mobilenetv2-coco_dr_1.tflite'
        get_category(img=file, model =model1 ) #saves output as image in static folder
        get_category(img=file, model =model2 )
        get_category(img=file, model =model3 )

        #from flask import Response
        return render_template('result.html', model1=model1, model2=model2, model3=model3)
        #Response(category.getvalue(), mimetype='image/png') 

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(port=33507, debug=True) #set to port 33507 so it runs in heroku
