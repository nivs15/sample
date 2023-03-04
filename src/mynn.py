import numpy as np
import pickle 

mod=pickle.load(open('myshallownetwork.pickle','rb'))
class MyFirstNN(object):
    def __init__(self):
        self.weights = mod['weights']
        self.bias = mod['bias']
    def sigmoid_function(self, x_input):
        return 1/(1 + np.exp(-x_input))
    def neuralnet(self,x_input):
        result = np.dot(x_input, self.weights) + self.bias
        res = self.sigmoid_function(result)
        return res
    def predict(self, x_input):
        x_input = np.array(x_input)
        pred = self.neuralnet(x_input)
        pred = list(pred)[0]
        if pred:
            pred=round(pred)
        else:
            pred=0
        pred={
            "predicted_class":pred
        }
        return pred
#x=MyFirstNN()
#input=[0,0]
#y=x.predict(input)
#print(y)
