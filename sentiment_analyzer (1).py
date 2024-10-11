from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

class SentimentAnalyser:
  def __init__(self):
    self.chat_model=ChatGoogleGenerativeAI(google_api_key='AIzaSyDbNeumk07ozAdTKYVppxdX3BjDBUItyJw',model="gemini-1.5-pro",temperature=0.5)
    self.prompt_generation()
    self.chain_creation()


  def prompt_generation(self):
    sentiment_template="""
    Given the user question below, classify which problem the user is facing:

    - Exam_pressure
    - Low_marks
    - Family_pressure
    - Performance_anxiety
    - Homesickness
    - Relationship_problems
    - Money_issues
    - Future_anxiety

    Do not respond with more than one word.
    Question:
    {sentiment_question}

    Classification:

    """
    self.prompt=PromptTemplate.from_template(template=sentiment_template,input_variable=['sentiment_question'] )


  def chain_creation(self):
    self.chain=LLMChain(
      llm=self.chat_model,
      prompt=self.prompt,
      output_parser=StrOutputParser(),
      output_key='response'
    )


  def run(self,query):
    output=self.chain.invoke({'sentiment_question':query})['response'].strip()
    return output
                                    



