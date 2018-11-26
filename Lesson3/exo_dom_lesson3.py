
# coding: utf-8

# In[2]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
from queue import Queue
from threading import Thread


# In[3]:


url = "https://gist.github.com/paulmillr/2657075"
api_url = "https://api.github.com"


# In[6]:


contributors = pd.read_html(url, attrs={"cellspacing":"0"})


# In[7]:


contributors


# In[10]:


most_active_users = contributors[0]


# In[12]:


most_active_users.head(5)


# In[ ]:


username = ["fabpot"]


# In[92]:



#import getpass

root_url='https://api.github.com/users/'
    
endpoint='/repos?page=1'
    
options='&per_page=500'

access_token = ''
    
headers = {'Authorization': 'token {}'.format(access_token)}
username = 'fabpot'

url=root_url+user+endpoint+options
r = requests.get(url,headers=headers)
json=r.json()

headers = {'Authorization': 'token {}'.format(access_token)}




# In[94]:



access_token = ''

def user_stars(user, access_token):
    root_url='https://api.github.com/users/'
    endpoint='/repos?page=1'
    options='&per_page=500'
    headers = {'Authorization': 'token {}'.format(access_token)}
    url=root_url+user+endpoint+options
    #print(url)
    r = requests.get(url,headers=headers)
    json=r.json()
    all_stars=[]
    for repos in json:
        all_stars+=[repos["stargazers_count"]]
    avr_stars = sum(all_stars)/len(all_stars) if len(all_stars) > 0 else 0
    #print(avr_stars)
    return avr_stars 

avr_stars = user_stars(username,access_token = '')
#f = response.json()
print(avr_stars)


# In[95]:


most_active_users['mean_stars'] = most_active_users['User'].apply(lambda u: user_stars(u.split(" ")[0], access_token = 'e7eb09512a1dd3b9c07d606a4f81bb65dc50a69f'))
most_active_users.head()


# In[97]:


top_star_rating = most_active_users.sort_values('mean_stars', ascending=False)

top_star_rating.head(5)

