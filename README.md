# RAG Application - Retrieval Augmented Generation

A powerful Retrieval Augmented Generation (RAG) system that allows you to chat with your documents using advanced AI models. The application provides both a web interface (Streamlit) and a command-line interface for interacting with your document collection.

## 🚀 Features

- **Document Processing**: Upload and process PDF documents for intelligent querying
- **AI-Powered Chat**: Interactive conversation interface with document-aware responses
- **Dual Interfaces**: Both web-based (Streamlit) and command-line interfaces
- **Vector Search**: Efficient document retrieval using FAISS vector indexing
- **Persistent Storage**: Document and vector embeddings are stored persistently
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Streaming Responses**: Real-time response generation with streaming support

## 🏗️ Architecture

The application uses a sophisticated RAG pipeline with the following components:

- **Document Loader**: Processes PDF files and extracts text content
- **Embeddings**: Creates vector embeddings for semantic search
- **Vector Store**: FAISS-based vector database for efficient retrieval
- **LLM Integration**: HuggingFace models for text generation
- **Graph Workflow**: LangGraph-based workflow for complex reasoning

## 📋 Prerequisites

### For Docker Setup
- Docker and Docker Compose installed
- HuggingFace account and API token

### For Local Setup
- Python 3.12 or higher
- Git (for cloning the repository)
- HuggingFace account and API token

## 🐳 Quick Start with Docker (Recommended)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag-app
```

### 2. Configure Environment Variables
Create a `.env` file from the example:
```bash
cp .env.example .env
```

Edit the `.env` file and add your HuggingFace token:
```bash
# HuggingFace API Token for model access
# Get your token from: https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=your_actual_huggingface_token_here

# Optional: Override default model
# MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
```

### 3. Run with Docker Compose
```bash
docker-compose up --build
```

### 4. Access the Application
- **Web Interface**: Open your browser and navigate to `http://localhost:8501`
- **Upload Documents**: Use the sidebar to upload PDF files
- **Start Chatting**: Type your questions in the chat interface

### 5. Stop the Application
```bash
docker-compose down
```

## 💻 Local Development Setup (Without Docker)

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd rag-app
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file:
```bash
cp .env.example .env
```

Edit the `.env` file with your HuggingFace token:
```bash
HUGGINGFACE_TOKEN=your_actual_huggingface_token_here
```

### 5. Run the Application

#### Option A: Web Interface (Streamlit)
```bash
streamlit run display.py
```
Then open `http://localhost:8501` in your browser.

#### Option B: Command Line Interface
```bash
python main.py
```

## 📁 Project Structure

```
rag-app/
├── main.py                 # Command-line interface
├── display.py              # Streamlit web interface
├── graph.py               # LangGraph workflow definition
├── llm.py                 # Language model configuration
├── embeddings.py          # Vector embedding utilities
├── document_loader.py     # PDF processing and loading
├── query.py               # Query processing logic
├── states.py              # Workflow state definitions
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker container configuration
├── docker-compose.yml    # Docker Compose setup
├── .env.example          # Environment variables template
├── documents/            # PDF document storage
├── context_index/        # Vector database storage
│   ├── index.faiss      # FAISS vector index
│   └── index.pkl        # Metadata storage
└── env/                 # Virtual environment (local setup)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `HUGGINGFACE_TOKEN` | HuggingFace API token for model access | Yes | - |
| `MODEL_NAME` | Override default model | No | `meta-llama/Llama-3.1-8B-Instruct` |

### Getting HuggingFace Token

1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with appropriate permissions
3. Copy the token to your `.env` file

## 📚 Usage

### Web Interface Features

1. **Document Upload**:
   - Use the sidebar file uploader
   - Upload PDF files to add to your knowledge base
   - Files are automatically processed and indexed

2. **Chat Interface**:
   - Type questions in the chat input
   - Receive AI-generated responses based on your documents
   - Conversation history is maintained during the session

3. **Real-time Responses**:
   - Responses are streamed in real-time
   - See the AI's thinking process as it generates answers

### Command Line Interface

1. **Interactive Mode**:
   ```bash
   python main.py
   ```

2. **Ask Questions**:
   - Enter your queries when prompted
   - Type 'exit' to quit the application

## 🔄 Adding New Documents

### Via Web Interface
1. Click the "Upload a PDF file to Add Context" in the sidebar
2. Select your PDF file
3. The document will be automatically processed and added to the knowledge base

### Via File System
1. Place PDF files in the `documents/` directory
2. Restart the application to reindex the documents

## 🛠️ Development

### Running Tests
```bash
# Add your test commands here when tests are implemented
```

### Code Structure

- **Graph Workflow**: The application uses LangGraph for orchestrating the RAG pipeline
- **Vector Storage**: FAISS is used for efficient similarity search
- **Streaming**: Real-time response generation with proper streaming support
- **State Management**: Persistent conversation state and document indexing

## 🐛 Troubleshooting

### Common Issues

1. **HuggingFace Token Error**:
   - Ensure your token is correctly set in the `.env` file
   - Verify the token has appropriate permissions

2. **Port Already in Use**:
   - Change the port in `docker-compose.yml` if 8501 is occupied
   - For local setup, Streamlit will automatically find an available port

3. **Memory Issues**:
   - Large PDF files may require more memory
   - Consider using a smaller model if running on limited resources

4. **Docker Build Issues**:
   - Ensure Docker daemon is running
   - Try `docker-compose down` and `docker-compose up --build`

### Performance Optimization

- **Vector Index**: The FAISS index is saved persistently to avoid recomputation
- **Document Caching**: Processed documents are cached for faster retrieval
- **Model Caching**: HuggingFace models are cached locally after first download

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended for larger models)
- **Storage**: 5GB free space (for models and documents)
- **CPU**: Multi-core processor recommended

### Recommended for Production
- **RAM**: 16GB or more
- **Storage**: SSD with 20GB+ free space
- **GPU**: CUDA-compatible GPU for faster inference (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the existing issues in the repository
3. Create a new issue with detailed information about your problem

## 🔮 Roadmap

- [ ] Support for more document formats (Word, PowerPoint, etc.)
- [ ] Multiple LLM provider support (OpenAI, Anthropic, etc.)
- [ ] Advanced query filters and search options
- [ ] User authentication and document isolation
- [ ] API endpoints for programmatic access
- [ ] Performance monitoring and analytics

---

**Happy RAG-ing! 🚀**
