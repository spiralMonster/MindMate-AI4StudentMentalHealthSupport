import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain_community.tools import TavilySearchResults

class RelaxationChain:
  def __init__(self,situation):
    self.chat_model=ChatGoogleGenerativeAI(google_api_key='AIzaSyDbNeumk07ozAdTKYVppxdX3BjDBUItyJw',model="gemini-1.5-pro",temperature=0.5)
    self.memory=ChatMessageHistory()
    self.situation=situation
    self.tool=TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False
    )

    self.prompt_generation()
    self.chain_creation()

  
  def prompt_generation(self):
    # Question Generator Prompt:
    temp1="""
    The user is under the stress of {situation}.
    You can cheer up the user by :
    - Cracking a Joke
    - Endulging user in some fun activity
    - Give some motivation
    Frame a question asking the user which of the above service would he/she like to enjoy
    """
    self.question_gen_prompt=PromptTemplate.from_template(input_variable=['situation'],template=temp1)

    # Activity Decider Prompt:
    temp2="""
    Given below is the user response and history of interaction between user and you.
    Decide what the user intends:
    - Joke
    - Fun_activity
    - Motivation
    - No_interest
    - Satisfied
    - Not_satisfied
    Your response should be one of the above
    Do not respond with more than one word.
    User response:
    {user}
    History of interaction:
    {history}
    """
    self.activity_decider_prompt=PromptTemplate.from_template(input_variable=['user','history'],template=temp2)

    # Joke Cracker Prompt:
    temp3="""
    The user is under the stress of {situation}.
    You have to cheer up the user by cracking a joke.
    Try to provide jokes related to exams.
    Use the following jokes as a context to get idea about what kind of joke to be provided:
    {jokes}
    """
    self.joke_prompt=PromptTemplate.from_template(input_variables=['situation','jokes'],template=temp3)

    # Motivation Prompt:
    temp4="""
    The user is under the stress of {}.
    You have to motivate the user.

    Use the following context to get idea about what kind of motivation is to be provided:
    {motivation}
    """

    self.motivation_prompt=PromptTemplate.from_template(input_variables=['situation','motivation'],template=temp4)

    # Help Seeker Prompt:
    temp5="""
    Looks like still the user is under stress.
    So now you have to ask user to seek medical attention.
    Provide future steps the user can take in order to overcome stress.
    Use the following context to get idea about how help is provided to overcome stress:
    {help}
    """
    self.help_prompt=PromptTemplate.from_template(input_variable=['help'],template=temp5)

  def chain_creation(self):
    #Question Generator Chain:
    self.question_gen_chain=self.question_gen_prompt|self.chat_model|StrOutputParser()

    #Activity Decider Chain:
    self.activity_decider_chain=LLMChain(
    llm=self.chat_model,
    prompt=self.activity_decider_prompt,
    output_parser=StrOutputParser(),
    output_key='response'
    )

    # Joke Chain:
    self.joke_chain=self.joke_prompt|self.chat_model|StrOutputParser()

    # Motivator Chain:
    self.motivator_chain=self.motivation_prompt|self.chat_model|StrOutputParser()

    #Help Seeker chain:
    self.help_chain=self.help_prompt|self.chat_model|StrOutputParser()

  #Context for jokes from Internet:
  def internet_joke(self):
    joke=self.tool.invoke({'query':'Give me some jokes about exams'})
    joke=[re.sub(r'\s\s+',' ',j['content']) for j in joke]
    return " ".join(joke)

    # Context for motivation from internet:
    def internet_motivation(self):
      motiv=self.tool.invoke({'query':'Give me some real life examples which were not good in academics but acheive a great success'})
      motiv=[re.sub(r'\s\s+',' ',j['content']) for j in motiv]
      return " ".join(motiv)

    # Context for professional help:
    def internet_help(self):
      help_context=self.tool.invoke({'query':'Platforms offering help to overcome stress.'})
      help_context=[
          f"""
          Url: {h['url']}
          Content:
          {h['content']}
          """
        for h in help_context
      ]
      help_context="\n".join(help_context)
      return re.sub(r'\s\s+',' ',help_context)
  
  def run(self):
    question=self.question_gen_chain.invoke({'situation':self.situation})
    print(f"AI: {question}")
    self.memory.add_ai_message(question)
    while True:
      user_resp=input("User: ")
      if user_resp=='exit':
        break

      activity=self.activity_decider_chain.invoke({'user':user_resp,'history':self.memory.messages})
      self.memory.add_user_message(user_resp)

      if activity['response'].strip()=='Joke':
        joke_context=self.internet_joke()
        joke=self.joke_chain.invoke({'situation':self.situation,'jokes':joke_context})
        print(f"AI: {joke}")
        self.memory.add_ai_message(joke)

      elif activity['response'].strip()=='Motivation':
        motivation_context=self.internet_motivation()
        motivation=self.motivator_chain.invoke({'situation':self.situation,'motivation':motivation_context})
        print(f"AI: {motivation}")
        self.memory.add_ai_message(motivation)


      elif activity['response'].strip()=='Not_satisfied' or activity['response'].strip()=='No_interest':
        help_context=self.internet_help()
        get_help=self.help_chain.invoke({'help':help_context})
        print(f"AI: {get_help}")
        self.memory.add_ai_message(get_help)

      else:
        break

if __name__=='main':
  relax=RelaxationChain(situation='exam pressure')
  relax.run()
