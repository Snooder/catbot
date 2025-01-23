from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.chains import SQLDatabaseChain
from langchain.llms import OpenAI
from sqlalchemy import create_engine

# Connect to a business database
engine = create_engine("sqlite:///business_data.db")
db = SQLDatabase(engine)

# Query the database using natural language
query = "What is the total revenue this month?"
response = db.run(query)
print(response)
