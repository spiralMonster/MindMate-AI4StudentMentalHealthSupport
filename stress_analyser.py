from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


class StressAnalyser:
    def __init__(self):
        self.chat_model = ChatGoogleGenerativeAI(google_api_key='AIzaSyDbNeumk07ozAdTKYVppxdX3BjDBUItyJw',
                                                 model="gemini-1.5-pro", temperature=0.5)
        self.prompt_generation()
        self.chain_creation()

    def prompt_generation(self):
        temp = """
    Given below is the history of interaction between the user and you.
    Decide whether the user is under the stress of exams or not.
    Give your response either as 'Stress' or 'No Stress'
    Do not respond with more than one word.
    History:
    {history}
    """
        self.prompt = PromptTemplate.from_template(input_variable=['history'], template=temp)

    def chain_creation(self):
        self.chain = LLMChain(
            llm=self.chat_model,
            prompt=self.prompt,
            output_parser=StrOutputParser(),
            output_key='response'
        )

    def run(self, history):
        output = self.chain.invoke({'history': history})['response'].strip()
        return output




