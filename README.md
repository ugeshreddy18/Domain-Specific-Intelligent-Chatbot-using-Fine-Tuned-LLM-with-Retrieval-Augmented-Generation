# Domain-Specific-Intelligent-Chatbot-using-Fine-Tuned-LLM-with-Retrieval-Augmented-Generation
Developed an AI-powered college admission assistant using Fine-Tuned LLM, Semantic Search, FAISS vector database, and Retrieval-Augmented Generation (RAG) for contextual question answering from uploaded PDF documents and common Query questions.

**🚀 Live Demo**

🌐 Live Application:
https://college-admission-chatbot.streamlit.app/

**📌 Features**

✅ Fine-Tuned LLM using LoRA

✅ Semantic Search for accurate responses

✅ Retrieval-Augmented Generation (RAG)

✅ PDF Upload and Question Answering

✅ FAISS Vector Database Integration

✅ Streamlit Professional UI

✅ Dark Mode Support

✅ Real-Time Chat Interface

✅ Smart Query Matching

✅ Responsive and Interactive Design

**🧠 Project Overview**

This chatbot is designed to assist students with college admission-related queries such as:
- Eligibility criteria
- Fee structure
- Scholarships
- Hostel facilities
- Placements
- Courses offered
- Contact information
- Admission process
- The system combines:

Fine-Tuned LLM responses
Semantic similarity search
PDF document retrieval using RAG to provide contextual and intelligent answers.

**🛠️ Technologies Used**

| Technology            | Purpose          |
| --------------------- | ---------------- |
| Python                | Core Programming |
| Streamlit             | Web Interface    |
| Transformers          | LLM Integration  |
| PEFT / LoRA           | Fine-Tuning      |
| Sentence Transformers | Semantic Search  |
| FAISS                 | Vector Database  |
| LangChain             | RAG Pipeline     |
| PyPDF2                | PDF Processing   |
| Hugging Face          | Model Loading    |


**🧩 System Architecture**

User Query
   ↓
Semantic Search
   ↓
Fine-Tuned LLM
   ↓
RAG PDF Retrieval
   ↓
   
Generated Response
📂 Project Structure
├── app.py
├── data.json
├── requirements.txt
├── fine_tuned_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json

**Screenshots**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4f66cdd1-9ec5-418b-9169-8ce72123d1cf" />

<img width="722" height="813" alt="image" src="https://github.com/user-attachments/assets/01ec9f67-fe4c-4002-952a-c83d6f78be15" />

**Dark mode feature:**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ca781f0c-6d96-4c84-b81c-9634173f602b" />

**📄 PDF RAG Functionality**

Users can upload PDF brochures/documents and ask contextual questions.
The chatbot:
1. Extracts PDF text
2. Splits into chunks
3. Generates embeddings
4. Stores vectors in FAISS
5. Retrieves relevant content
6. Generates accurate answers

**🎨 UI Features**

- Modern Chat Interface
- Dark Mode Toggle
- Suggested Questions
- Chat History
- Clear Chat Option
- Animated Cards
- Responsive Layout

**📈 Future Enhancements**

- Voice Assistant Integration
- User Authentication
- Multi-PDF Support
- Database Chat Storage
- Advanced LLM Models
- Analytics Dashboard
- Multilingual Support

**👨‍💻 Developed By**

Borra Ugesh Reddy

ML Internship Project 🚀
