#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder

class MulticolumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        output = X.copy()
        for col in self.columns:
            output[col] = self.encoders[col].transform(output[col])
        return output

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# In[ ]:




