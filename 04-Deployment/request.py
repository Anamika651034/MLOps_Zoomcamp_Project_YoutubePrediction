import requests
import pandas as pd

new_data = {
     "VIEWS": "50.98B",
      "TOTAL_NUMBER_OF_VIDEOS": 799,
      "CATEGORY": "Entertainment",
   }

#new_data = pd.DataFrame({'VIEWS': [1000000], 'TOTAL_NUMBER_OF_VIDEOS': [500], 'CATEGORY': ['Entertainment']})


url = 'http://localhost:5001/api'
r = requests.post(url,json=new_data)
print(r.text)