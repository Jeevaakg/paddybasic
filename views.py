
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
        
models = keras.models.load_model('E:/Projects/Paddy Leaf Detecion/Deploy/model.h5')



test_image = image.load_img('E:/Projects/Paddy Leaf Detecion/Deploy/media/images/brown_spot_01.PNG',
                            target_size=(224, 224))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

result = models.predict(test_image)

prediction = result[0]

prediction = list(prediction)

classes = ['Brown Spot', 'Healthy Leaf', 'Leaf Blast', 'Leaf Blight', 'Leaf Smut']

output = zip(classes, prediction)

output = dict(output)

#print(output)

if output['Brown Spot'] == 1.0:
    a='Brown Spot'
elif output['Healthy Leaf'] == 1.0:
    a='Healthy Leaf'
elif output['Leaf Blast'] == 1.0:
    a="Leaf Blast"
elif output['Leaf Blight'] == 1.0:
    a="Leaf Blight"
elif output['Leaf Smut'] == 1.0:
    a="Leaf Smut"

print(a)
