import os
import re
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import LLMMathChain
from langchain.tools import DuckDuckGoSearchRun, tool

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0)
search_tool = DuckDuckGoSearchRun()

@tool
def pdf_parser(self):
    """Reads and parses credit card statement"""
    loader = PyPDFLoader("statement.pdf")
    pages = loader.load_and_split()

    # Regular expression pattern for transaction extraction (thanks chatgpt for this lol)
    pattern = re.compile(r'(\d{2} [A-Z]{3})\s*\n\s*([^0-9\n]+)\s*\n\s*R?\$? ?([0-9.,]+)')

    transactions = []
    for page in pages[3:]:  # First 3 pages doesn't contain transactions (Nubank)
        for match in re.finditer(pattern, page.page_content):
            date, merchant, amount = match.groups()
            transactions.append({
                "Date": date,
                "Merchant Name": merchant.strip(),
                "Amount": amount.replace(',', '')
            })

    return transactions

@tool
def category_knowledge(self):
    """Pulls past transaction merchants and their spending categories"""
    loader = CSVLoader(file_path="categories.csv")
    data = loader.load()
    return data

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

@tool
def calculation(query):
    """Calculates values. The query needs the operation writen with the numbers to make the calculation."""
    llm_math = LLMMathChain.from_llm(llm, verbose=True)
    return llm_math.invoke(query)

# Define your agents with roles and goals
labeler = Agent(
  role='Financial Transaction Labeler',
  goal='Label financial transactions from credit card statement based on spending category',
  backstory="""You work at the best AI financial labeling company.
  Your expertise lies in identifying what the category of spending in each transaction from a credit card statement.""",
  verbose=True,
  allow_delegation=True,
  tools=[pdf_parser,search_tool,category_knowledge]     
)
presenter = Agent(
  role='Financial Presenter',
  goal='Present the credit card spending based on each category',
  backstory="""You are a senior financial results presenter
  You get the transactions from a credit card statement and present them based on each category""",
  verbose=True,
  allow_delegation=True,
  tools=[calculation] 
  # (optional) llm=ollama_llm
)

statement = Task(
    description="""Organize and Return the full content of the card statement as follows: Date, Mechant Name, Amount.
    Ignore transactions named "Pagamento em" and "Estorno" because these are not spendings, but payments I received.
    You MUST print ALL transactions no matter how long the list is.
    """,
    tools=[pdf_parser],
    agent=labeler
)
# This is only needed if you want to make it more accurate. Have an csv name categories.csv
merchant_category = Task(
    description="Pull merchant categories",
    tools=[category_knowledge],
    agent=labeler
)

unknown_categories = Task(
    description="Search for unknown merchant categories. Query with only the name of the merchant",
    tools=[search_tool],
    agent=labeler
)

categorizer = Task(
    description="""Categorize each transaction from the credit card statement with it's merchant spending category using the previous tasks knowledge.
        There are 2 category spendings: Fixed and Variable.
        Inside Fixed transactions there are these subcategories: Utilities, Rent, Car Insurance and Health Insurance.
        Inside Variable transactions there are these subcategories: Groceries, Restaurants, Subscription, Shopping, Donation, Farmacy, Gas, Car Toll, Travel and Other.
        You MUST go through all the pages in the PDF and each transaction needs to have the date, category, subcategory and amount.
        This is the expected output format:
        1. Date, Mechant Name, Category, Subcategory, Amount
        2. Date, Mechant Name, Category, Subcategory, Amount
        3. Date, Mechant Name, Category, Subcategory, Amount
        ... 
        """,
    agent=labeler
)

total_spending = Task(
    description="""Get the total spending cost. Use the previous tasks knowledge and the calculation tool. The input on the tool needs to be the full operation expression""",
    agent=presenter,
    tools=[calculation]
)

total_fixed_spending = Task(
    description="""Get the total fixed spending cost and the percentage representation from the total_spending amount. Use the previous tasks knowledge and the calculation tool. The input on the tool needs to be the full operation expression""",
    agent=presenter,
    tools=[calculation]
)

total_variable_spending = Task(
    description="""Get the total fixed spending cost and the percentage representation from the total_spending amount. Use the previous tasks knowledge and the calculation tool. The input on the tool needs to be the full operation expression""",
    agent=presenter,
    tools=[calculation]
)

fixed_subcategory_spending = Task(
    description="""Get each subcategory total spending cost and the percentage representation from the total_fixed_spending amount. Use the previous tasks knowledge and the calculation tool. The input on the tool needs to be the full operation expression""",
    agent=presenter,
    tools=[calculation]
)

variable_subcategory_spending = Task(
    description="""Get each subcategory total spending cost and the percentage representation from the total_variable_spending amount. Use the previous tasks knowledge and the calculation tool. The input on the tool needs to be the full operation expression""",
    agent=presenter,
    tools=[calculation]
)

present = Task(
  description="""Using amounts, categories and subcategories provided, present the information in a clear and insighfull way.

  This is how you should present the data:

  Total: <total_spending_in_month>

  Fixed: <total_fixed_cost>, <fixed_percentage_of_total_cost>
  Variable: <total_variable_cost>, <variable_percentage_of_total_cost>

  Fixed Cost Top 5 Biggest Spendings:
  1. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_fixed_cost>
  2. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_fixed_cost>
  3. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_fixed_cost>
  4. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_fixed_cost>
  5. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_fixed_cost>

  Variable Cost Top 5 Biggest Spendings:
  1. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_varibale_cost>
  2. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_varibale_cost>
  3. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_varibale_cost>
  4. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_varibale_cost>
  5. <subcategory_name>: <total_amount_sub_category>, <subcategory_name_of_total_varibale_cost>

  Add a one paragraph long conclusion in the end giving insights on the spending""",
  agent=presenter
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[labeler, presenter],
  tasks=[statement, 
         merchant_category, 
         unknown_categories, 
         categorizer, 
         total_spending,
         total_fixed_spending,
         total_variable_spending,
         fixed_subcategory_spending,
         variable_subcategory_spending,
         present],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)