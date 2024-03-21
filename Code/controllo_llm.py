import openai
# -----------------
# Global variables
# -----------------
MODEL = "gpt-3.5-turbo"
openai.api_key = "sk-INSERIREKEY"

def build_messages(system_prompt, user_prompt, assistant_prompt):
    """
        Builds the messages to be sent to the LLM.
    """
    messages = []

    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": assistant_prompt})
    return messages

def controllo_llm(transazione):
    SYSTEM_PROMPT = "You're a financial companion and fraud detector \n\
    User provides a transaction, and based on your knowledge you have to guess if it is a genuine transaction or a fraudolent transaction \n\
    Answer using two sentences explaining the outcome and why you chose it as outcome. You also need to check if the transaction makes sense."
    
    
    USER_PROMPT= f"\n\
    Amount: {transazione['Amount']}\n\
    Type: {transazione['Type']}\n\
    Old balance of the origin: {transazione['OldBalanceOrig']}\n\
    New balance of the origin: {transazione['NewBalanceOrig']}\n\
    Old balance of the destination: {transazione['OldBalanceDest']}\n\
    New balance of the destination: {transazione['NewBalanceDest']}"
    
    ASSISTANT_PROMPT = '''Use the following format:\n\
    Hello, i'm your Financial Assistant and I will assist you to understand this transaction.
    Your transaction is likely to be <your outcome>\n\
    The reason for the outcome is: <your explanation>'''

    messages = build_messages(SYSTEM_PROMPT, USER_PROMPT, ASSISTANT_PROMPT)

    response = openai.ChatCompletion.create(
        model = MODEL,
        messages=messages,
        temperature=0,
    )
    
    return response.choices[0].message.content
