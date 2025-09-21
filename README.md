# TriLink - AI-Powered Telecom Customer Service Analytics

TriLink is an innovative AI-powered customer service analytics platform that combines telecom services (mobile, internet) with home security solutions. The project demonstrates how BigQuery's multimodal AI capabilities can analyze customer interactions, predict churn, and generate intelligent cross-selling recommendations for a hybrid telecom + home security company.

## 🌟 What TriLink Does

TriLink simulates a comprehensive telecom company that offers:
- **Mobile Services**: Plans, data packages, device management
- **Internet Services**: Fiber, broadband, business connectivity
- **Home Security**: Smart cameras, doorbells, monitoring systems

The platform uses Google BigQuery's AI features to:
- Analyze customer call transcripts and support tickets
- Predict customer churn risk using machine learning
- Generate personalized retention offers and cross-selling opportunities
- Process multimodal data (text, images, customer profiles)
- Provide real-time agent guidance during customer calls

## 🎯 Key Features

### AI-Powered Customer Analytics
- **Multimodal Analysis**: Combines call transcripts, customer data, and device images
- **Vector Search**: Find similar customer cases and successful resolution patterns
- **Churn Prediction**: ML models to identify at-risk customers
- **Real-time Recommendations**: Generate personalized offers during live calls

### Comprehensive Data Generation
- **Realistic Customer Profiles**: Demographics, service history, usage patterns
- **Call Transcripts**: Authentic customer service conversations
- **Support Tickets**: Multi-sector problem-solution database
- **Device Images**: Smart home and security equipment visuals

### Advanced Business Intelligence
- **Cross-selling Optimization**: Identify upsell opportunities across service lines
- **Customer Lifetime Value**: Calculate and optimize customer retention value
- **Market Differentiation**: Unique telecom + security service combination

## 📁 Repository Structure

```
TriLink/
├── src/                         # Data simulation and generation modules
│   ├── calltranscripts_simulation.py    # Generate realistic customer service call transcripts
│   ├── create_customer_data.py          # Create customer profiles (100K customers with patterns)
│   ├── create_service_data.py           # Generate internet, mobile, security service data
│   ├── create_images.py                 # Create issue ticket images using AI
│   └── utils.py                         # Utility functions for data processing
├── queries/                     # BigQuery SQL templates for Agent tools
│   └── querystore.py                   # Centralized query management for ADK agents
├── trilink_notebooks/           # Main Jupyter notebooks (execution order)
│   ├── create_solution_embeddings.ipynb # Generate customer problem-solution data & embeddings
│   ├── create_models.ipynb              # Set up BigQuery ML models and views
│   ├── bigquery_actions.ipynb           # Core AI analytics workflows & demonstrations
│   └── describe-product-images-with-bigframes-multimodal.ipynb  # Multimodal analysis
├── data/                        # Pre-generated dataset storage
│   ├── images/                  # Device and product images for tickets
│   └── *.csv                    # Customer, call, and service data (ready to use)
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Google Cloud Platform account with BigQuery access
- Jupyter Notebook or JupyterLab
- Google Cloud SDK (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hazardscarn/TriLink
   cd TriLink
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Cloud authentication**
   ```bash
   # Install Google Cloud SDK
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   
   # Or set up service account key
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
   ```

### Configuration

1. **Update project settings** in the notebooks:
   - Open `trilink_notebooks/create_models.ipynb`
   - Change `project_id='trilink-472019'` to your Google Cloud project ID
   - Optionally update `database_name` and `location` variables

2. **BigQuery setup**:
   - Ensure BigQuery API is enabled in your GCP project
   - Create a dataset named `database` (or update the database_name variable)
   - Grant necessary permissions for BigQuery ML and AI features

## 🎮 Running the Project

### Quick Start (Using Pre-generated Data)
**By default, you don't need to run any data simulation.** All datasets are pre-generated and available in the `data/` folder. Simply follow these steps in order:

### Step 1: Create Simulated Datasets (Setup)
```bash
jupyter notebook trilink_notebooks/create_simulated_datasets.ipynb
```
This notebook:
- Sets up the BigQuery environment and database
- Loads pre-generated customer data from `data/` folder
- Creates all necessary tables (customers, internet, mobile, security, calls)
- Uploads datasets to BigQuery for analysis

### Step 2: Generate Solution Embeddings (Setup)
```bash
jupyter notebook trilink_notebooks/create_solution_embeddings.ipynb
```
This notebook:
- Creates customer problem-solution datasets across mobile, internet, and home security sectors
- Generates embeddings for similarity search and pattern matching
- Sets up vector search capabilities for business solutions
- Prepares data for AI-powered recommendations

### Step 3: Create ML Models and Views (Setup)
```bash
jupyter notebook trilink_notebooks/create_models.ipynb
```
This sets up:
- BigQuery ML models for churn prediction (mobile, internet, security)
- Customer segmentation and clustering algorithms
- Data views for efficient analysis and querying
- Feature engineering pipelines for model training

### Step 4: Run AI Analytics (Main Demo)
```bash
jupyter notebook trilink_notebooks/bigquery_actions.ipynb
```
**This is the main demonstration notebook** that showcases:
- Real-time customer analysis during calls
- Churn risk assessment and prediction
- Personalized offer generation and retention scripts
- Cross-selling recommendation engine
- Agent performance analytics
- ADK-powered agent tools and workflows

## 🔄 Advanced: Data Simulation (Optional)

If you want to generate fresh datasets or understand the data creation process, the `src/` folder contains simulation modules:

### Custom Data Generation
```python
# Generate 100K customers with realistic patterns
from src.create_customer_data import generate_customer_data
customers_df = generate_customer_data(num_customers=100000)

# Create service data (internet, mobile, security)
from src.create_service_data import generate_internet_services_fixed
internet_services_data = generate_internet_services_fixed(customers_df)

# Generate realistic call transcripts
from src.calltranscripts_simulation import create_enhanced_trilink_calls
calls_df = create_enhanced_trilink_calls(
    customers_df, internet_df, mobile_df, security_df,
    num_calls=1000
)

# Create device/issue images using AI
from src.create_images import generate_ticket_images
images = generate_ticket_images(ticket_data)
```

**Note:** The simulation modules are used internally by the notebooks when needed. Most users should start with the pre-generated data in the `data/` folder.


## 🛠 Technical Stack

- **Google BigQuery**: Data warehouse and AI/ML platform
- **BigQuery ML**: Machine learning model training and inference
- **Vertex AI**: Advanced AI capabilities and multimodal analysis
- **Google ADK (Agent Development Kit)**: Agent orchestration and tool integration
- **Python**: Primary programming language
- **Google Cloud Storage**: File and image storage


## 🔧 Customization


### Generating Fresh Data
The `src/` folder contains powerful simulation engines:
- **`calltranscripts_simulation.py`**: Creates realistic customer service conversations using AI
- **`create_customer_data.py`**: Generates 100K customers with insightful demographic patterns
- **`create_service_data.py`**: Creates comprehensive service data across all sectors
- **`create_images.py`**: Generates device issue images using AI image generation

### Agent Query Management
The `queries/` folder provides centralized SQL template management:
- **`querystore.py`**: Contains all BigQuery SQL templates used by ADK agents
  - Customer retention script generation
  - Churn prediction queries
  - Vector search for similar problems
  - Technician report queries
  - Agent performance analytics

### Adjusting AI Models
- Modify prompts and analysis logic in BigQuery SQL
- Tune ML model parameters and features in `create_models.ipynb`
- Update vector search similarity thresholds
- Customize business rules and recommendations

### Extending Multimodal Capabilities
- Add new image types and analysis functions
- Integrate additional data sources (audio, video)
- Enhance device recognition and troubleshooting
- Expand object table schemas

## 📝 License

This project is for demonstration and educational purposes. Please ensure compliance with your organization's data privacy and AI usage policies.

## 🤝 Contributing

This is a demonstration project showcasing BigQuery AI capabilities. Feel free to fork and extend for your own use cases.

## 📞 Support

For questions about the BigQuery AI features used in this project, refer to:
- [BigQuery ML Documentation](https://cloud.google.com/bigquery-ml/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [BigQuery Multimodal Analysis](https://cloud.google.com/bigquery/docs/multimodal-analysis)

---

*TriLink demonstrates the power of combining traditional  services with modern AI analytics to create comprehensive customer experiences and drive business growth.*