
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 



from xgboost import XGBClassifier

model_fname = "model.save"
MODEL_NAME = "bin_class_base_xgb_lime"


class Classifier(): 
    
    def __init__(self, n_estimators=250, eta=0.3, gamma=0.0, max_depth=5, **kwargs) -> None:
        self.n_estimators = n_estimators
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.model = self.build_model(**kwargs)     
        self.train_X = None
        
    def build_model(self, **kwargs): 
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            eta=self.eta,
            gamma=self.gamma,
            max_depth=self.max_depth,
            verbosity = 0, **kwargs, )
        return model
    
    
    def fit(self, train_X, train_y):    
        self.train_X = train_X     
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        return self.model.predict(X) 
    
    
    def predict_proba(self, X):
        return self.model.predict_proba(X) 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        model = joblib.load(os.path.join(model_path, model_fname))
        return model


def save_model(model, model_path):
    model.save(model_path)
 

def load_model(model_path): 
    model = joblib.load(os.path.join(model_path, model_fname))   
    return model


