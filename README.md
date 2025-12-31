# Smart Water Quality Analyzer

A Streamlit-based web application that analyzes water quality using a hybrid approach combining World Health Organization (WHO) rule-based safety checks, machine learning predictions, and AI-generated expert insights.

## Project Overview

This project implements a comprehensive water quality assessment tool that:

- **WHO Rule-Based Checks**: Enforces strict safety thresholds for pH, turbidity, conductivity, dissolved oxygen, and TDS based on WHO guidelines.
- **Machine Learning Prediction**: Uses a Gradient Boosting Classifier trained on water quality data to provide probabilistic safety assessments.
- **AI Insights**: Leverages Google's Gemini AI to generate detailed expert analysis, treatment recommendations, and usage suggestions.

The hybrid approach ensures that WHO rules act as hard fails, while ML provides nuanced predictions for borderline cases, and AI offers practical engineering advice.

## Features

- **Interactive Parameter Input**: User-friendly interface for entering water quality parameters (pH, Turbidity, Conductivity, Dissolved Oxygen, TDS).
- **Synced Inputs**: Automatic synchronization between Conductivity and TDS values using a conversion factor.
- **Safety Assessment**: Real-time classification of water as Safe, Unsafe, or requiring treatment.
- **ML Confidence Scores**: Displays probability scores from the machine learning model.
- **AI-Generated Insights**: Structured expert reports including:
  - Water quality classification
  - Key issues identified
  - Recommended treatment methods
  - Post-treatment usage possibilities
  - Health and environmental considerations
- **Responsive Design**: Centered layout optimized for various screen sizes.

## Installation

### Prerequisites

- Python 3.8 or higher
- A Google Gemini API key (obtain from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Setup Steps

1. **Clone or download the project files** to your local machine.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv water_quality_env
   source water_quality_env/bin/activate  # On Windows: water_quality_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**:
   - Create a `.env` file in the project root directory.
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

5. **Ensure model files are present**:
   - `gb_water_model.pkl` (trained Gradient Boosting model)
   - `scaler.pkl` (feature scaler)
   - These should be included in the project files.

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the app**:
   - Open your web browser and navigate to the URL displayed (typically `http://localhost:8501`).

3. **Enter water parameters**:
   - Input values for pH, Turbidity (NTU), Conductivity (µS/cm), Dissolved Oxygen (mg/L), and TDS (ppm).
   - Note: TDS and Conductivity are automatically synced.

4. **Analyze**:
   - Click the "🔍 Analyze Water Quality" button.
   - View the safety assessment and ML confidence score.
   - Read the AI-generated expert insights and recommendations.

## Model Training

The project includes a training script (`model.py`) that creates the ML model used for predictions.

### Training Process

1. **Data Loading**: Loads water quality data from `Water Quality Testing.csv`.
2. **Preprocessing**:
   - Filters data for temperature range (20-30°C).
   - Selects features: pH, Turbidity, Conductivity, Dissolved Oxygen, TDS.
   - Handles missing values.
3. **Labeling**: Applies WHO rule-based labeling (0: Unsafe, 1: Safe).
4. **Model Training**: Trains a Gradient Boosting Classifier with 300 estimators.
5. **Evaluation**: Prints classification report on test set.
6. **Artifact Saving**: Saves model (`gb_water_model.pkl`) and scaler (`scaler.pkl`).

### Retraining the Model

To retrain with new data:

1. Update `Water Quality Testing.csv` with new samples.
2. Run the training script:
   ```bash
   python model.py
   ```
3. The updated model and scaler will overwrite the existing files.

## Dependencies

- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `joblib`: Model serialization
- `python-dotenv`: Environment variable management
- `google-genai`: Google Gemini AI integration

## Dataset

- **File**: `Water Quality Testing.csv`
- **Purpose**: Training data for the machine learning model
- **Features**: pH, Turbidity (NTU), Conductivity (µS/cm), Dissolved Oxygen (mg/L), TDS (ppm), Temperature (°C)
- **Note**: The model is trained on data filtered for 20-30°C temperature range.

## File Structure

```
water-quality-analyzer/
├── app.py                 # Main Streamlit application
├── model.py               # ML model training script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
├── gb_water_model.pkl     # Trained Gradient Boosting model
├── scaler.pkl             # Feature scaler
└── Water Quality Testing.csv  # Training dataset
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## Disclaimer

This application is for educational and informational purposes only. It is not a substitute for professional water quality testing or expert consultation. Always consult with qualified water treatment specialists for critical applications. The AI-generated insights are advisory and should not be used as operational instructions without proper validation.

---
