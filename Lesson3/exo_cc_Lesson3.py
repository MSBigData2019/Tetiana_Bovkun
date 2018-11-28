
# coding: utf-8

# In[3]:


from bs4 import BeautifulSoup
import requests
import re
import json


# In[4]:


import pandas as pd
villes = pd.read_html("https://fr.wikipedia.org/wiki/Liste_des_communes_de_France_les_plus_peupl%C3%A9es", header=0)[0]


# In[5]:


villes.head(50)


# In[6]:


def get_distance(a,b):
    r = requests.get("https://fr.distance24.org/route.json?stops="+a+"|"+b)
    json_ = json.loads(r.text)
    return json_["distance"]


# In[69]:


get_distance("Paris", "Marseille")


# In[96]:


cities = pd.DataFrame(city.split('[')[0] for city in villes.Commune[:50])


# In[97]:


cities.shape


# In[100]:


cities.iloc[49,0]

    
#cities


# In[102]:


Distance_Map = pd.DataFrame(index = cities, columns = cities)
#y = df.iloc[:,-1:]
for i in range (0 , (len(cities) - 1)):
    for j in range (0 , (len(cities) - 1)):
        if (i == j):
            Distance_Map.iloc[i,i] = 0
        else:
            Distance_Map.iloc[i,j] = Distance_Map.iloc[j,i] = get_distance(cities.iloc[i,0],cities.iloc[j,0])
    


# In[104]:


Distance_Map.head(5)

