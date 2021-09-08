from mlforkids import MLforKidsImageProject

# treat this key like a password and keep it secret!
key = "59fdafa0-0344-11ec-b2a1-2fd72466e147ed3ad9ca-5a6a-48da-984e-3cf57d577df0"

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction("carboat/"+input())

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))