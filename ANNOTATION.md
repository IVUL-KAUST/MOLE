# FORM

first you need to 
`git clone https://github.com/zaidalyafeai/mole_form` and `pip install -r requirements.txt`
then start the server using `streamlit run app.py`
The annotation guidlines are [here](https://docs.google.com/document/d/1m6fesR0-VO2cWK1O_FOPlj7f-NZn0UjdWYG0jGWIp9s/edit?usp=sharing). Before starting the annotation you need first to change the schema to annotate a paper for it. Add the arXiv pdf to start the AI annotation for example https://arxiv.org/pdf/2211.05958. If the dataset is already annotated using this model it will preload the annotation. 

# Backend

`git clone https://github.com/IVUL-KAUST/MOLE` and change the branch to `molev2` then `pip install -r requirements.txt`
then start the server using `uvicorn main:app`. Make sure you add .env with `OPENROUTER_API_KEY=sk**`