import os

import streamlit as st
from streamlit import session_state as ss
from dotenv import load_dotenv
from openai.types.beta.assistant_stream_event import ThreadMessageDelta, ThreadRunRequiresAction, \
    ThreadMessageInProgress, ThreadMessageCompleted, ThreadRunCompleted
from openai.types.beta.threads.text_delta_block import TextDeltaBlock
import json
import time
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import re

# openai variables
load_dotenv()
model = 'gpt-4o-mini'
client = OpenAI()

def create_assistant():
    print('CREATING NEW ASSISTANT')
    print('...creating a vectorstore')
    # Create a vector store caled "Financial Statements"
    vector_store = client.beta.vector_stores.create(name="guides datafiles 1",
        chunking_strategy = {
            'type': 'static',
            'static': {
                'max_chunk_size_tokens': 800,
                'chunk_overlap_tokens': 400
            }
        }
    )
    # Ready the files for upload to OpenAI
    FILES_TO_UPLOAD = ['GUIDET_Becoming_Entrepreneur_in_Finland.pdf']
    file_streams = [open(path, "rb") for path in FILES_TO_UPLOAD]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    print('...file status:')
    print('......' + str(file_batch.status))
    print('......' + str(file_batch.file_counts))

    print('...creating assistant')
    my_assistant = client.beta.assistants.create(
        description='Smart guides assistant',
        instructions='You are an honest and smart assistant that helps immigrants and enterpreneurs in Finland. You only respond questions related to those topics.\nYou must provide a precise answer to a query using your knowledge. If answer cannot be found in knowledge, say you don\'t know. Don\'t make up an answer.\nStrictly follow these rules: \n1. Analyze the query to provide only the precise answer. Do not provide any additional information unless necessary and helpful in the query\'s context.\n2. You do not need to provide the answer exactly in the same words as in the context/information. You may style the answer in appropriate format and words. \nFor example:\nquery = "query: What are the working hours in Finland?"\ncontext = "weekly working time should not exceed 40 hours"\nresponse = "The working hours in Finland are 40 hours per week."\n3. Consider creating bullet points or bold headings/sub-headings or other formatting as necessary.',
        model='gpt-4o-mini',temperature = 0.05,name='Smart-Business-Guide (API)',
        tools=[{"type": "file_search"},{'type':'function','function': {
                   'name': 'get_tax_rates',
                   'description': 'Get the current tax rates in Finland',
                   'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False, 'required': []},
                   'strict': True
               }},
               {'type':'function','function': {
                   'name': 'get_migri_contacts',
                   'description': 'Get contact information of immigration services (migri) in Finland',
                   'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False, 'required': []},
                   'strict': True
               }}
               ],
    )
    print('...adding vectorstore to assistant')
    my_assistant = client.beta.assistants.update(
        assistant_id=my_assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    print('...done!')
    return my_assistant

try:
    assistant_id = os.getenv('ASSISTANT_ID')
    my_assistant = client.beta.assistants.retrieve(assistant_id)
    print('using existing assistant')
except:
    my_assistant = create_assistant()
    os.environ["ASSISTANT_ID"]=my_assistant.id
    assistant_id = os.getenv('ASSISTANT_ID')
    print(f'created new assistant with ID {assistant_id}')

def get_assistant_summary():
    tools = my_assistant.tools
    print('...getting assistant summary info')
    if 1: # get names through vectorstores
        databases = [x for x in my_assistant.tool_resources if x[0]=='file_search']
        vector_store_ids = list(*[x[1].vector_store_ids for x in databases])
        vector_store_objects = [client.beta.vector_stores.files.list(vector_store_id=id) for id in vector_store_ids]
        vector_store_files = [y.id for y in list(*[x.data for x in vector_store_objects])]
        filenames = [client.files.retrieve(x).filename for x in vector_store_files]
    else: # get all files, might not be part of assistant!
        filenames = [x.filename for x in client.files.list()]
    assistant_summary = f'functions ({len(tools)}):'+'  \n'+'  \n'.join(['-'+x.function.description for x in tools if (x.type=='function')]) + '  \n' + f'files ({len(filenames)}): ' + '; '.join(['"'+x+'"' for x in filenames])
    print('...done!')
    return assistant_summary

tax_rates_url = r'https://www.vero.fi/en/businesses-and-corporations/taxes-and-charges/vat/rates-of-vat/'
migri_contacts_url = r'https://migri.fi/en/contact-information'

def remove_tags(soup):
    # Remove unwanted tags
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        element.decompose()

    # Extract text while preserving structure
    content = ""
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        text = element.get_text(strip=True)
        if element.name.startswith('h'):
            level = int(element.name[1])
            content += '#' * level + ' ' + text + '\n\n'  # Using Markdown-style headings
        elif element.name == 'p':
            content += text + '\n\n'
        elif element.name == 'li':
            content += '- ' + text + '\n'

    return content

def get_html(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to retrieve agencies")
    soup = BeautifulSoup(response.text, "html.parser")
    text = remove_tags(soup)
    return text

def execute_required_function(func_name, arguments):
    if func_name == "get_tax_rates":
        return get_html(tax_rates_url)
    if func_name == "get_migri_contacts":
        return get_html(migri_contacts_url)

class Assistant:
    thread_id = ""

    def __init__(self, model: str = model):
        # openai variables
        self.client = client
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None
        self.summary = None

        # retrieve existing assistant based on hardcoded data
        self.assistant = self.client.beta.assistants.retrieve(
            assistant_id=assistant_id
        )

        # create thread as this initialization only occurs on boot up of app
        if Assistant.thread_id:
            self.thread = self.client.beta.threads.retrieve(
                thread_id=Assistant.thread_id
            )
        else:
            self.thread = self.client.beta.threads.create()
            Assistant.thread_id = self.thread.id

    def add_user_prompt(self, role, content):
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content
            )

    def stream_response(self, assistant_reply_box):
        try:
            with client.beta.threads.runs.create(
                    assistant_id=self.assistant.id,
                    thread_id=self.thread.id,
                    stream=True
            ) as stream:
                assistant_reply = ""
                start_time = time.time()
                max_duration = 120  # Maximum duration in seconds for streaming

                # Iterate through the stream of events
                for event in stream:
                    print("Event received!")  # Debug statement

                    # Check if the maximum duration has been exceeded
                    if time.time() - start_time > max_duration:
                        print("Stream timeout exceeded.")
                        break

                    # Handle different types of events
                    if isinstance(event, ThreadMessageDelta):
                        print("MSG ThreadMessageDelta event data")  # Debug statement
                        if isinstance(event.data.delta.content[0], TextDeltaBlock):
                            # add the new text
                            assistant_reply += event.data.delta.content[0].text.value
                            # display the new text
                            assistant_reply_box.markdown(assistant_reply)

                    elif isinstance(event, ThreadRunRequiresAction):
                        print("ThreadRunRequiresAction event data")  # Debug statement

                        # Get required actions
                        runs_page = self.client.beta.threads.runs.list(thread_id=self.thread.id)
                        runs = list(runs_page.data)
                        if runs:
                            run = runs[0]
                            run_id = run.id if hasattr(run, 'id') else None

                            if run_id:
                                required_actions = run.required_action.submit_tool_outputs.model_dump()
                                tool_outputs = []

                                # Loop through actions
                                for action in required_actions["tool_calls"]:
                                    # Identify function and params
                                    func_name = action["function"]["name"]
                                    arguments = json.loads(action["function"]["arguments"])
                                    print(
                                        f"Executing function: {func_name} with arguments: {arguments}")  # Debug statement

                                    # Run the agent function caller
                                    output = execute_required_function(func_name, arguments)
                                    print(f"Function {func_name} complete")  # Debug statement

                                    # Create the tool outputs
                                    tool_outputs.append({"tool_call_id": action["id"], "output": str(output)})

                                # Submit the outputs
                                if tool_outputs:
                                    print("Tool output acquired")
                                    with client.beta.threads.runs.submit_tool_outputs(
                                            thread_id=self.thread.id,
                                            run_id=run_id,
                                            tool_outputs=tool_outputs,
                                            stream=True
                                    ) as stream:
                                        print("Streaming response to tool output...")
                                        # Handle different types of events
                                        for event in stream:
                                            if isinstance(event, ThreadMessageDelta):
                                                print("TOOL ThreadMessageDelta event data")  # Debug statement
                                                if isinstance(event.data.delta.content[0], TextDeltaBlock):
                                                    # add the new text
                                                    assistant_reply += event.data.delta.content[0].text.value
                                                    # display the new text
                                                    assistant_reply_box.markdown(assistant_reply)

                    elif isinstance(event, ThreadMessageInProgress):
                        print("ThreadMessageInProgress event received")  # Debug statement
                        time.sleep(1)

                    elif isinstance(event, ThreadMessageCompleted):
                        print("Message completed.")  # Debug statement

                    elif isinstance(event, ThreadRunCompleted):
                        print("Run completed.")  # Debug statement

                    print("Loop iteration completed.")  # Debug statement to check loop progress

                return assistant_reply

        except Exception as e:
            print("An error occurred during streaming: ", str(e))
            return "An error occurred while processing your request."

# Initialize agent
if 'agent' not in ss:
    ss.agent = Assistant()
    ss.initial_message_shown = False
    ss.chat_history = []

# Streamlit app configuration
st.set_page_config(
    page_title="Smart Business Guide",
    page_icon="✈️",
)

# App title
st.title("Smart guides")

# Display initial message if not shown before
if not ss.initial_message_shown:
    assistant_summary = get_assistant_summary()
    initial_message = "Hello! I'm an entrepreneurship assistant. I have following tools in use:\n\n"+ assistant_summary + "\n\nHow can I assist you today?"
    ss.initial_message_shown = True
    ss.chat_history.append({"role": "assistant", "content": initial_message})

# Display chat messages from history on app rerun
for message in ss.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about entrepreneurship in Finland"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to app chat history
    ss.chat_history.append({"role": "user", "content": prompt})

    # Send message to chatbot
    ss.agent.add_user_prompt("user", prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Empty container to display the assistant's reply
        assistant_reply_box = st.empty()

        # Initialize the assistant reply as an empty string
        assistant_reply = ""

        # Stream the assistant's response
        assistant_reply = ss.agent.stream_response(assistant_reply_box)

        # Once the stream is over, update chat history
        ss.chat_history.append({"role": "assistant", "content": assistant_reply})

