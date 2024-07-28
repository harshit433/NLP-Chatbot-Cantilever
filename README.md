# ChefAssist: NLP-Based AI Chatbot for Culinary Haven

ChefAssist is an AI chatbot designed to serve as a customer service agent for Culinary Haven. The chatbot responds to customer queries regarding various aspects of the restaurant, such as the name of the chatbot, the name of the restaurant, timings, menu, accessibility, parking, items, payment modes, and more.

## ScreenShot
![Screenshot 2024-07-28 225602](https://github.com/user-attachments/assets/3689f7ce-df2e-47cd-9a59-4c6c3f2d17d5)


## Project Directory Structure

```
.
├── dataset
│   ├── chatbot-data.json
│   └── words.py
├── Model
│   └── model.pth
├── app.py
├── predictions.py
├── static
│   └── css
├── templates
│   └── chat.html
```

### Description of Files and Directories

- **dataset/**
  - `chatbot-data.json`: Contains the dataset used for training the chatbot.
  - `words.py`: Contains utility functions for processing text data.

- **Model/**
  - `model.pth`: The trained model for the chatbot.

- **app.py**: The main application file for running the Flask web server.

- **predictions.py**: Contains the code for making predictions using the trained model.

- **static/css**: Directory for static CSS files.

- **templates/chat.html**: HTML template for the chatbot user interface.

## Setup and Installation

### Prerequisites

- Python 3.x
- Flask
- PyTorch
- NLTK

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**

   Create a virtual environment and install the required packages:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Download NLTK Data**

   If not already downloaded, install necessary NLTK data:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

4. **Run the Application**

   Start the Flask application:

   ```bash
   python app.py
   ```

5. **Access the Chatbot**

   Open your web browser and navigate to `http://127.0.0.1:5000` to interact with ChefAssist.

## How It Works

1. **Data Processing**: The `words.py` script processes the text data from `chatbot-data.json`.
2. **Model Training**: The chatbot model is trained using the data and stored in `model.pth`.
3. **Prediction**: The `predictions.py` script uses the trained model to make predictions based on user queries.
4. **Flask Application**: The `app.py` file runs a Flask web server that serves the chatbot interface and handles user interactions.

## Future Enhancements

- Improve the chatbot's response accuracy.
- Add more features to handle a wider range of customer queries.
- Integrate with a live restaurant database for real-time updates.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
