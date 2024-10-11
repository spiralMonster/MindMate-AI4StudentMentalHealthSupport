import os
import re
import asyncio
from textwrap import dedent

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import (
    PromptTemplate, ChatPromptTemplate,
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory
from langchain_community.tools import TavilySearchResults

from sentiment_analyzer import SentimentAnalyser
from stress_analyser import StressAnalyser
from relaxation import RelaxationChain

os.environ["TAVILY_API_KEY"] = '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'


class ExamPressureHandlerAI:
    def __init__(self):
        self.chat_model = ChatGoogleGenerativeAI(google_api_key='AIzaSyDbNeumk07ozAdTKYVppxdX3BjDBUItyJw',
                                                 model="gemini-1.5-pro", temperature=0.5)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                                            google_api_key='AIzaSyDbNeumk07ozAdTKYVppxdX3BjDBUItyJw')
        self.sentiment_analyzer = SentimentAnalyser()
        self.stress_analyzer = StressAnalyser()
        self.relaxation_chain = RelaxationChain(situation='exam pressure')
        self.tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        self.memory = ChatMessageHistory()
        self.prompt_generation()
        self.chain_creation()

    def embed_data_vectorstore(self):

        # Text Splitter:
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)

        # Data Sources:
        web_data_sources = ['https://www.ucl.ac.uk/news/2017/apr/7-tips-help-you-cope-exam-stress',
                            'https://www2.hse.ie/mental-health/life-situations-events/exam-stress/',
                            'https://www.concordia.ca/cunews/offices/provost/health/topics/stress-management/exam-stress.html',
                            'https://kidshelpline.com.au/teens/issues/exam-stress',
                            'https://www.qld.gov.au/youth/looking-after-your-mental-health/managing-your-thoughts/exam-stress',
                            'https://www.studentminds.org.uk/examstress.html',
                            'https://www.nhs.uk/mental-health/children-and-young-adults/help-for-teenagers-young-adults-and-students/tips-on-preparing-for-exams/']

        video_data_sources = ['-RZ86OB9hw4', 'H1TqVWEZOs4']

        # Data Transformation:
        web_loader = WebBaseLoader(web_data_sources)
        data_web = web_loader.load()
        data_web = [re.sub(r'\s\s+', ' ', d.page_content) for d in data_web]
        data_video = [re.sub(r'\s\s+', ' ', YoutubeLoader(src).load()[0].page_content) for src in video_data_sources]
        data = []
        data.extend(data_web)
        data.extend(data_video)
        split_data = [splitter.split_text(inst) for inst in data]
        final_data = []
        for data in split_data:
            for text in data:
                final_data.append(text)

        # Vectorstore:
        self.vectorstore = Chroma.from_texts(final_data, self.embedding_model)

    # Format text:
    def format_text(self, result):
        data = [re.sub(r'\s\s+', ' ', d.page_content) for d in result]
        return " ".join(data)

    async def get_context(self, query):
        results = await self.vectorstore.asearch(query, search_type='mmr', k=3)
        final_results = self.format_text(results)
        return final_results

    def prompt_generation(self):
        # Prompt whether we are dealing with same problem or not:
        temp1 = """
    Depending upon the user query decide whether the query is about exams or not.
    Give your response either as 'Yes' or 'No'.
    Do not respond with more than one word.
    User Query:
    {query}
    """
        self.problem_checker_prompt = PromptTemplate.from_template(input_variable=['query'], template=temp1)

        # Prompt whether context is required or not:
        context_req_sys_temp = """
    Depending upon the user query you will be provided the context which will include the information about:
      - How to prepare for exams?
      - How to handle exam pressure?
      - How to help the students who are in stress due to exam pressure?
      - How to manage the stress during exams?

    But in questions like:
    - How to prepare for Physics exam?
    - What topics are important for maths exam?
    The context would not be helpful. So you have to use Internet Serach to get answers to these questions.
    """
        context_req_temp = """
    Depending upon the user query decide whether the context is required or the Internet Search to answer the query.
    Your response should be either 'Context' or 'Internet_search'
    Do not respond with more than one word.
    User query:
    {query}
    """
        self.context_req_prompt = ChatPromptTemplate(
            messages=[SystemMessagePromptTemplate.from_template(context_req_sys_temp),
                      HumanMessagePromptTemplate.from_template(context_req_temp)],
            input_variables=['query']
        )

        # Question-Answering Prompt:
        sys_mess_temp = """
    The user is under the stress as his/her exams are around the corner.
    You are helpful assitant which will help the user in the following ways:
      - Provide proper techniques for exam preparation.
      - Help the user to manage the stress.
      - Create a roadmap for the exam preparation.
      - Ensure the proper time management for the preparation.
      - Motivate the user.
      - Help to reduce the stress of the user by cheering the user.
    """

        prom_temp = """
    The user query:
    {question}
    Use the following as a context to solve the query of the user:
    {context}
    """

        self.question_answering_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(sys_mess_temp),
                HumanMessagePromptTemplate.from_template(prom_temp)
            ],
            input_variables=['question', 'context']

        )

    def chain_creation(self):
        # Chain to decide whether the problem is same or not.
        self.problem_checker_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.problem_checker_prompt,
            output_parser=StrOutputParser(),
            output_key='response'
        )

        # Chain to provide context or not:
        self.context_req_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.context_req_prompt,
            output_parser=StrOutputParser(),
            output_key='response'
        )

        # Question Answering Chain:
        self.question_answering_chain = self.question_answering_prompt | self.chat_model | StrOutputParser()

    def run(self, initial_question):
        # First flag so that the initial question is served as the second question and the user dont'have to type query again.
        flag_first = True
        # Second flag so that when exited from the relaxation chain the sentiment is determined again.
        flag_second = False

        sentiment = None
        num_of_queries = 0
        while True:
            if (num_of_queries + 1) % 5 == 0:

                print("**Checking whether the user is in stress or not")
                print("----------------------------------------------")
                stress = self.stress_analyzer.run(history=self.memory.messages)

                if stress == 'Stress':
                    print("**The user is in stress**")
                    print("---------------------------------------")
                    print("**Activating Relaxation chain: **")
                    print("---------------------------------------")
                    return_question = self.relaxation_chain.run()
                    flag_second = True

                else:
                    print("**The user is not in stress**")
                    print("---------------------------------------")
                    print("**Executing our current chain**")
                    print("---------------------------------------")

            if flag_first:
                question = initial_question
                flag_first = False
                num_of_queries += 1

            elif flag_second:
                question = return_question
                flag_second = False

            else:
                question = input("User: ")
                num_of_queries += 1

            if question == 'exit':
                break

            same_chain = self.problem_checker_chain.invoke({'query': question})

            if same_chain['response'].strip() == 'Yes':
                print("**Process is going on in same Process**")
                print("---------------------------------------")
                context_req = self.context_req_chain.invoke({'query': question})

                if context_req['response'].strip() == 'Context':
                    print("**The context from vectorstore is used**")
                    print("---------------------------------------")
                    context = self.get_context(question)
                    answer = self.question_answering_chain.invoke({'question': question, 'context': context})
                    print(f"AI: {answer}")
                    self.memory.add_user_message(question)
                    self.memory.add_ai_message(answer)

                else:
                    print("**The context is used from internet**")
                    print("---------------------------------------")
                    search = self.tool.invoke({'query': question})
                    search = [re.sub(r'\s\s+', ' ', d['content']) for d in search]
                    search = "\n".join(search)
                    answer = self.question_answering_chain.invoke({'question': question, 'context': search})
                    print(f"AI: {answer}")
                    self.memory.add_user_message(question)
                    self.memory.add_ai_message(answer)

            else:
                print("**Process has exited from chain**")
                print("---------------------------------------")
                sentiment = self.sentiment_analyzer.run(query=question)
                print(f"**Process now entered in {sentiment} chain.**")
                print("---------------------------------------")
                break

        return self.memory.messages, sentiment
