# T-NLP
## Handle the verbosity of error
#from transformers.utils import logging
#logging.set_verbosity_error()
from transformers import pipeline
chatbot = pipeline(task="conversational", model="./models/facebook/blenderbot-400M-distill")

user_message = """
What are some fun activities I can do in the summber?
"""

from transformers import Conversation
conversation = Conversation(user_message)
print(conversation)

conversation = chatbot(conversation)
print(conversation)

conversation.add_message(
    {"role":"user", 
     "content":"""
        What else do you recommend?
        """
    })
print(conversation)

conversation = chatbot(conversation)
print(conversation)
