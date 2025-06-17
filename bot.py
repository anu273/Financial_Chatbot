import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from htmlTemplate import css, bot_template, user_template
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import spacy
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import io
import base64
import matplotlib.pyplot as plt

load_dotenv()

def get(endpoint, params=None):
    base_url = f'https://finnhub.io/api/v1/{endpoint}'
    if not params:
        params = {}
    params['token'] = os.getenv("FINNHUB_API_KEY")
    response = requests.get(base_url, params=params)
    return response.json()

def find_stock_symbol(company_name):
    results = get('search', {'q': company_name})
    
    if 'result' in results and results['result']:
        top_result = results['result'][0]
        company_full_name = top_result['description']
        symbol = top_result['symbol']
        return symbol, company_full_name
    else:
        return None, None
    

def get_financial_data(symbol, start_date, end_date):
    # Helper function for safe nested access
    def get_safe(data, *keys, default='N/A'):
        try:
            for key in keys:
                data = data[key]
            return data
        except (KeyError, IndexError, TypeError):
            return default

    # Fetch data from Finnhub
    profile = get('stock/profile2', {'symbol': symbol}) or {}
    quote = get('quote', {'symbol': symbol}) or {}
    news = get('company-news', {'symbol': symbol, 'from': start_date, 'to': end_date}) or []
    income = get('stock/financials-reported', {'symbol': symbol}) or {}
    earnings = get('stock/earnings', {'symbol': symbol}) or []
    recommendations = get('stock/recommendation', {'symbol': symbol}) or []
    technical = get('indicator', {
        'symbol': symbol,
        'resolution': 'D',
        'from': int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()),
        'to': int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()),
        'indicator': 'rsi',
        'timeperiod': 14
    }) or {}
    insider = get('stock/insider-transactions', {'symbol': symbol}) or {}
    sentiment = get('stock/social-sentiment', {'symbol': symbol}) or {}
    esg = get('stock/esg', {'symbol': symbol}) or {}
    peers = get('stock/peers', {'symbol': symbol}) or []

    # Extract income report data safely
    ic_data = income.get("data", [{}])[0].get("report", {}).get("ic", {}) if income.get("data") else {}
    revenue = ic_data.get("Revenue", "N/A")
    net_income = ic_data.get("NetIncome", "N/A")

    # Build summary
    summary = f"""
📊 **Financial Summary for `{symbol}`**

🧾 **Company Profile**:
- Name: {profile.get('name', 'N/A')}
- Exchange: {profile.get('exchange', 'N/A')}
- Industry: {profile.get('finnhubIndustry', 'N/A')}
- Website: {profile.get('weburl', 'N/A')}

💰 **Real-Time Quote**:
- Current Price: {quote.get('c', 'N/A')}
- Open: {quote.get('o', 'N/A')}
- High: {quote.get('h', 'N/A')}
- Low: {quote.get('l', 'N/A')}
- Previous Close: {quote.get('pc', 'N/A')}

📈 **Technical Indicator (RSI)**:
- RSI Values: {(technical.get('rsi', [])[:5] if isinstance(technical.get('rsi', []), list) else 'N/A')}

🧾 **Income Statement**:
- Total Revenue: {revenue}
- Net Income: {net_income}

📆 **Earnings**:
{''.join([f"- {e.get('period', 'N/A')}: EPS = {e.get('actual', 'N/A')}, Estimate = {e.get('estimate', 'N/A')}\n" for e in earnings[:3]]) if isinstance(earnings, list) and earnings else 'N/A'}

📊 **Analyst Recommendations**:
{''.join([f"- Period: {r.get('period', 'N/A')}, Buy: {r.get('buy', 'N/A')}, Hold: {r.get('hold', 'N/A')}, Sell: {r.get('sell', 'N/A')}\n" for r in recommendations[:2]]) if isinstance(recommendations, list) and recommendations else 'N/A'}

🔍 **Insider Transactions**:
{''.join([f"- {i.get('name', 'N/A')}: {i.get('transactionType', 'N/A')} {i.get('share', 'N/A')}\n" for i in insider.get('data', [])[:3]]) if isinstance(insider.get('data', []), list) and insider.get('data') else 'N/A'}

💬 **Social Sentiment**:
- Reddit Mentions: {sentiment.get('reddit', [{}])[0].get('mention', 'N/A') if sentiment.get('reddit') else 'N/A'}
- Twitter Mentions: {sentiment.get('twitter', [{}])[0].get('mention', 'N/A') if sentiment.get('twitter') else 'N/A'}

🌱 **ESG Scores**:
- Environment Score: {esg.get('environmentScore', 'N/A')}
- Social Score: {esg.get('socialScore', 'N/A')}
- Governance Score: {esg.get('governanceScore', 'N/A')}

👥 **Peers**:
{', '.join(peers) if isinstance(peers, list) and peers else 'N/A'}

📰 **Recent News**:
{''.join([f"- {n.get('datetime', '')}: {n.get('headline', '')}\n" for n in news[:3]]) if isinstance(news, list) and news else 'N/A'}
"""

    # Store in session state with better organization
    st.session_state.financial_context = {
        'summary': summary,
        'symbol': symbol,
        'period': f"{start_date} to {end_date}",
        'raw_data': {
            "profile": profile,
            "quote": quote,
            "news": news,
            "income": income,
            "earnings": earnings,
            "recommendations": recommendations,
            "technical": technical,
            "insider": insider,
            "sentiment": sentiment,
            "esg": esg,
            "peers": peers
        }
    }

    # Store in memory with a standardized format
    if "memory" not in st.session_state:
        st.session_state.memory = got_memory()
        
    memory = st.session_state.memory
    
    
    # Save the detailed financial context to memory for future retrieval
    memory.save_context(
        {"question": f"Get financial data and analysis for {symbol} from {start_date} to {end_date}"},
        {"answer": summary}
    )
    
    # Also store the raw data in session state for backward compatibility
    st.session_state['financial_data'] = st.session_state.financial_context['raw_data']
    st.session_state['financial_summary'] = summary
    
    return summary


def get_stock_data(company_name, start_date, end_date, interval):
    # Find stock symbol (assuming find_stock_symbol is defined elsewhere)
    ticker_info = find_stock_symbol(company_name)
    if not ticker_info:
        st.error("Could not find stock symbol for the provided company name.")
        return
    symbol, full_name = ticker_info

    yf_ticker = yf.Ticker(symbol)  # Use yfinance to get stock data
    try:
        # Fetch stock data
        stock_data = yf_ticker.history(start=start_date, end=end_date, interval=interval)
        if stock_data.empty:
            error_msg = f"No stock data found for {full_name} ({symbol}) in the specified date range."
            st.session_state.chat_history.append(AIMessage(content=f"❌ {error_msg}"))
            return
        
        # Save to session state
        st.session_state.last_stock_data = stock_data
        st.session_state.last_stock_symbol = symbol
        st.session_state.last_company_name = full_name

        # Prepare Excel download link
        excel_buffer = io.BytesIO()
        # Remove timezone info from index if present
        if stock_data.index.tz is not None:
            stock_data = stock_data.copy()
            stock_data.index = stock_data.index.tz_localize(None)
        stock_data.to_excel(excel_buffer)
        excel_buffer.seek(0)
        b64_excel = base64.b64encode(excel_buffer.read()).decode()
        download_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{symbol}_stock_data.xlsx">Download data</a>'

        # Display stock data in chat
        summary = f"""
📈 Retrieved stock data for {full_name} (`{symbol}`) from {start_date} to {end_date} with interval - {interval}.

**Stock Data:**
{stock_data.to_markdown()}

![Stock Price Chart](data:image/png;base64,{_get_chart_as_base64(stock_data)})

{download_link}
"""
        # Save to both chat history and memory
        st.session_state.chat_history.append(AIMessage(content=summary))
        if 'memory' not in st.session_state:
            st.session_state.memory = got_memory() 
        memory = st.session_state.memory
        memory.save_context(
            {"question": f"Show stock data for {company_name}"},
            {"answer": summary}
        )

    except Exception as e:
        error_msg = f"Error fetching stock data: {str(e)}"
        st.session_state.chat_history.append(AIMessage(content=f"❌ {error_msg}"))
        st.error(error_msg)

def _get_chart_as_base64(stock_data):
    """Helper function to convert chart to base64 for embedding in markdown"""
    
    plt.figure(figsize=(10, 4))
    plt.plot(stock_data.index, stock_data['Close'])
    plt.title('Stock Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_pdf_text(pdf_docs):
    file_contents = []
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            file_contents.append({
                "filename": pdf.name,
                "pages": len(reader.pages),
                "text": text,
                "size": f"{pdf.size / 1024:.1f} KB"
            })
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
            file_contents.append({
                "filename": pdf.name,
                "error": str(e)
            })
    return file_contents
    

def get_text_chunks(file_contents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks_with_metadata = []
    for file_info in file_contents:
        if "text" in file_info:
            chunks = text_splitter.split_text(file_info["text"])
            for chunk in chunks:
                chunks_with_metadata.append({
                    "text": chunk,
                    "source": file_info["filename"],
                    "pages": file_info["pages"],
                    "size": file_info["size"]
                })
    
    return chunks_with_metadata


def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Separate texts and metadata
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [{"source": chunk["source"], 
                 "pages": chunk["pages"],
                 "size": chunk["size"]} for chunk in text_chunks]
    
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectorstore

def got_memory():
    # Function to initialize memory for the conversation
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
        )
    return st.session_state.memory

def get_conversation_history(vectorstore):
    template = """
You are a Senior Financial Analyst with experience in corporate finance, equity research, and investment banking.

Respond based on the availability of context from these three sources:
---

📄 **Document Content** (Earnings Calls, Filings, Reports):
{context}

📊 **Live Financial Summary** (Real-time metrics, news, financials):
{supplementary}

🧠 **Conversation History**:
{chat_history}

---

### 🔍 Instructions:

1. **If only live financial data is available**, do the following:
   - Focus on performance metrics, trends, and analyst-style summary.
   - Avoid referencing missing document data.
   - Use this structure:
     - Executive Summary
     - Financial Highlights
     - Technical/Market Commentary
     - Recommendation
     - Risk/Reward Outlook
     - News highlights
     - Key Metrics

2. **If only document data is available**:
   - Focus on textual cues (CEO quotes, earnings mentions, forecasts).
   - Summarize key insights and potential investor implications.
   - Avoid guessing real-time performance.

3. **If both are available**, cross-validate:
   - Confirm document insights against financial metrics.
   - Spot inconsistencies, highlight confirmations or contradictions.
   - Provide risk/reward outlook using both sources.
   - Relate document insights to real-time performance.

4. **If neither are available**:
   - Respond as a general financial advisor.
   - Offer helpful insights or interpretations based on question alone.

### ✍️ Current Question:
{question}

---

Answer in a professional, insightful, and well-structured tone. Use bullet points or sections if necessary. Always cite which source your insight came from when applicable.
"""

    llm = OllamaLLM(model="llama3.2", temperature=0.3)

    # Setup memory (session-based)
    if "memory" not in st.session_state:
        memory = got_memory()
    else:
        memory = st.session_state.memory

    # Create conversation history chain
    conversation_history = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template=template,
                input_variables=["context", "supplementary", "chat_history", "question"]
            )
        },
        output_key="answer",
        return_source_documents=True
    )

    return conversation_history


def handle_user_input(user_question):
    if not user_question:
        return

    try:
        if 'memory' not in st.session_state:
            st.session_state.memory = got_memory()
        memory = st.session_state.memory

        # Prepare financial context if available
        financial_context = st.session_state.get('financial_context', {})
        supplementary_data = ""
        if financial_context:
            supplementary_data = f"""
**Current Financial Analysis Context:**
Symbol: {financial_context.get('symbol', 'N/A')}
Period: {financial_context.get('period', 'N/A')}

{financial_context.get('summary', '')}
"""

        llm = OllamaLLM(model="llama3.2", temperature=0.3)

        # Use PDF/vectorstore if available
        if st.session_state.conversation:
            try:
                response = st.session_state.conversation.invoke({
                    "question": user_question,
                    "supplementary": supplementary_data,
                    "financial_data": financial_context
                })
                answer = response.get("answer", "No answer found.")


            except Exception as e:
                # Fall back to general LLM if vectorstore fails
                error_note = f"⚠️ PDF analysis failed: {str(e)}\n\nSwitching to general model..."
                context_prompt = f"""Financial Context: {supplementary_data}\n\nQuestion: {user_question}"""
                llm_response = llm.invoke(context_prompt)
                answer = f"{error_note}\n\n🧠 LLM Response:\n{llm_response}"

        elif 'financial_context' in st.session_state and financial_context:
            # No vectorstore, directly use LLM with financial context if any
            context_prompt = f"""Financial Context: {supplementary_data}\n\nQuestion: {user_question}"""
            answer = llm.invoke(context_prompt)

        else:
            answer = llm.invoke(user_question)
        
        # Save to memory and chat history
        memory.save_context(
            {"question": user_question},
            {"answer": answer}
        )
        st.session_state.chat_history.append(AIMessage(content=answer))

    except Exception as e:
        error_msg = f"❌ Error processing your question: {str(e)}"
        st.session_state.chat_history.append(AIMessage(content=error_msg))
        memory.save_context(
            {"question": user_question},
            {"answer": error_msg}
        )

    # Clear input field
    st.session_state.user_question = ""


def main():
    load_dotenv()  # Load environment variables from .env file 
    st.set_page_config(page_title="Financial ChatBot", page_icon=":bar_chart:", layout="wide")  
    st.markdown(css, unsafe_allow_html=True)  # Replace this
  

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "input_question" not in st.session_state:
        st.session_state.input_question = ""
    if "memory" not in st.session_state:
        st.session_state.memory = got_memory()
    if "financial_context" not in st.session_state:
        st.session_state.financial_context = {}

    st.title("Financial Analyst AI Chatbot")


    # Display chat history
    with st.container():
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        
    # Use a form to capture Enter key press
    with st.form(key="main_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_input(
                "Ask your question or enter stock symbol:",
                key="input_question",
                label_visibility="collapsed",
                placeholder="Type your question or stock symbol here..."
            )
        with col2:
            submitted_chat = st.form_submit_button("Enter")

    if submitted_chat and user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        handle_user_input(user_input)
        st.rerun()
    

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], key="file_uploader", accept_multiple_files=True)
        if st.button("Upload", key="load_docs"):
            if not pdf_docs:  # Check if no file is uploaded
                st.error("No File Uploaded. Please upload at least one PDF file.")
            else:
                with st.spinner("Loading documents..."):
                    # Extract text from the pdf files
                    raw_text = get_pdf_text(pdf_docs)

                    # Create text chunks from the raw text
                    text_chunks = get_text_chunks(raw_text)

                    # Create vectorstore
                    vectorstore = get_vectorstore(text_chunks)
                    st.success("Uploaded successfully!")

                    #Conversation history
                    st.session_state.conversation = get_conversation_history(vectorstore)
        
        st.write("")

        st.subheader("Stock Data")
        if 'company_name' not in st.session_state:
            st.session_state.company_name = ""

        company_name = st.text_input("Company Name")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        time_interval = st.selectbox(
            "Time Interval",
            options=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            key="time_interval"
        )
        submit = st.button("Get Stock Data")

        if submit:
            if not company_name:
                st.error("Please enter a company name.")
            else:
                st.session_state.company_name = ""
                question = f"Show stock data for {company_name} company from {start_date} to {end_date} with interval {time_interval}."
                st.session_state.chat_history.append(HumanMessage(content=question))

                # Save stock request in session state and rerun
                st.session_state.pending_stock_request = {
                    "company_name": company_name,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "interval": time_interval,
                }
                st.session_state.memory.save_context(
                    {"question": question},
                    {"answer": "Fetching stock data..."}
                )
                symbol, company = find_stock_symbol(company_name)
                get_stock_data(company_name, str(start_date), str(end_date), time_interval)
                get_financial_data(symbol, str(start_date), str(end_date))
                st.session_state.chat_history.append(AIMessage(content=f"Fetched financial data for {company_name} ({symbol}) from {start_date} to {end_date}. You can ask related questions!"))
                st.rerun()
            

if __name__ == "__main__":
    main()