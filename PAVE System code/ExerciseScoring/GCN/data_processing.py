import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
from IPython.core.debugger import set_trace




index_Head=1   # no orientation (no.1, bp 0, nose)
index_Shoulder_Left=46 # (no.2, bp 11, left_shoulder)
index_Elbow_Left=54 # (no.3, bp 13, left_elbow)
index_Wrist_Left=62 # (no.4, bp 15, left_wrist)
index_Hand_Left=70 # (no.5, bp 18, left_pinky)
index_Shoulder_Right=50 # (no.6, bp 12, right_shoulder)
index_Elbow_Right=58 # (no.7, bp 14, elbow_right)
index_Wrist_Right=66 # (no.8, bp 16, right_wrist)
index_Hand_Right=74 # (no.9, bp 18, right_pinky)
index_Hip_Left=94 # (no.10, bp 23, left_hip)
index_Knee_Left=102 # (no.11, bp 25, left_knee)
index_Ankle_Left=110 # (no.12, bp 27, left_ankle)
index_Foot_Left=126  # no orientation # (no.13, bp 31, left_foot_index)  
index_Hip_Right=94 # (no.14, bp 24, right_hip)
index_Knee_Right=106 # (no.15, bp 26, right_knee)
index_Ankle_Right=114 # (no.16, bp 28, right_ankle)
index_Foot_Right=130   # no orientation # (no.17, bp 32, right_foot_index)

index_Tip_Left=78     # no orientation (no.18, bp 19, left_index)
index_Thumb_Left=86   # no orientation (no.19, bp 21, left_thumb)
index_Tip_Right=82    # no orientation (no.20, bp 20, right_index)
index_Thumb_Right=90  # no orientation (no.21, bp 22, right_index)

class Data_Loader():
    def __init__(self, dir):
        self.num_repitation = 5
        self.num_channel = 3
        self.dir = dir
        self.body_part = self.body_parts()       
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 100
        self.new_label = []
        self.train_x, self.train_y= self.import_dataset()
        self.batch_size = self.train_y.shape[0]
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.scaled_x, self.scaled_y = self.preprocessing()
                
    def body_parts(self):
        body_parts = [index_Head, index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
]
        return body_parts
    
    def import_dataset(self):
        train_x = pd.read_csv("./" + self.dir+"/Train_X.csv", header = None).iloc[:,:].values
        train_y = pd.read_csv("./" + self.dir+"/Train_Y.csv", header = None).iloc[:,:].values
        return train_x, train_y
            
    def preprocessing(self):
        X_train = np.zeros((self.train_x.shape[0],self.num_joints*self.num_channel)).astype('float32')
        for row in range(self.train_x.shape[0]):
            counter = 0
            for parts in self.body_part:
                for i in range(self.num_channel):
                    X_train[row, counter+i] = self.train_x[row, parts+i]
                counter += self.num_channel 
        
        y_train = np.reshape(self.train_y,(-1,1))
        X_train = self.sc1.fit_transform(X_train)         
        y_train = self.sc2.fit_transform(y_train)
        
        X_train_ = np.zeros((self.batch_size, self.num_timestep, self.num_joints, self.num_channel))
        
        for batch in range(X_train_.shape[0]):
            for timestep in range(X_train_.shape[1]):
                for node in range(X_train_.shape[2]):
                    for channel in range(X_train_.shape[3]):
                        X_train_[batch,timestep,node,channel] = X_train[timestep+(batch*self.num_timestep),channel+(node*self.num_channel)]
        
                        
        X_train = X_train_                
        return X_train, y_train


