# ProjetWEM JVC

Create virtual environment
```
python -m venv venv
```

Activate the virtual environment:
```
venv\Scripts\activate # Windows
source venv/bin/activate # MacOS and Linux
```

Then install all the packages in the `requirements.txt` file.
```
pip install -r requirements.txt
```

Launch crawler `crawler-selenium.py`.

```
python crawler-selenium
```

Launch streamlit server `streamlit_app.py`.

```
streamlit run streamlit_app.py
```

Code to get data from .csv `/data/dataset500.csv`.

```python
from tools.read import get_data

# Load data
dataset = get_data('data/dataset500.csv')

# Select your features
data = [{'game': dataset['name'][i], 'platform': dataset['platform'][i],
         'grade': comment['grade'][0], 'comment': comment['comment'][0], 'username': comment['username'][0]}
        for i, comments in enumerate(dataset['comments']) for comment in comments]

# Create a dataframe
df = pd.DataFrame(data)
```

Code for Review Bombing Detection.

Require a dataframe, the confidence index ('Low, 'Medium', 'High'), the sentiment ('positive', 'negative') and the game's name.

```python
from review_bombing import review_bombing_prediction_process

review, pred = review_bombing_prediction_process(dataframe=df,
                                                 confidence='Low',
                                                 sentiment='positive',
                                                 game='Gran Turismo 7')
```

