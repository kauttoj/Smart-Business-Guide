# Smart-Business-Guide
Smart Business Guide
A RAG application with streamlit to query business/entrepreneurship guides and get precise answers. OPENAI API key is requierd to use the guide.  
With no PDF guide loaded, the smart guide works as ChatGPT.  
After uploading a PDF guide, the application takes a few seconds to compute its embedding and save it to a vector database in local directory. This database is available in next runs.  
Run the app using the following command:  
`streamlit run smart_guide_OpenAI.py`  or `python -m streamlit run smart_guide_OpenAI.py`


