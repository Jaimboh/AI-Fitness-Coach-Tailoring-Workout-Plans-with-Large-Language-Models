# AI Fitness Coach: Tailoring Workout Plans with Large Language Models

This project implements an AI-powered Fitness Coach that creates personalized fitness plans using Large Language Models (LLMs). It leverages the LangChain and LangGraph frameworks to create a multi-agent system that interacts with users, generates fitness plans, and provides ongoing support and motivation.

## Features

- User profile creation based on input data
- Personalized fitness plan generation
- Feedback collection and plan adjustment
- Progress monitoring
- Motivational support
- Interactive streamlit UI for easy user interaction

## Caution

**Important Note:** While this AI Fitness Coach aims to provide personalized fitness advice, it should not be considered a substitute for professional medical or fitness expertise. The generated plans are based on general fitness principles and may not account for all individual health conditions or circumstances. Always consult with a healthcare professional or certified fitness trainer before starting any new exercise regimen, especially if you have pre-existing health conditions or injuries.

## Requirements

- Python 3.7+
- LangChain
- LangGraph
- Streamlit
- Ollama (for running LLMs locally)
- LangChain Community
- Langchain OpenAI
- OpenAI (optional, for cloud-based models)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Jaimboh/AI-Fitness-Coach-Tailoring-Workout-Plans-with-Large-Language-Models
   cd AI-Fitness-Coach-Tailoring-Workout-Plans-with-Large-Language-Models
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```
   ```
   source venv/bin/activate
   ```
   # On Windows  use
   ```
   venv\Scripts\activate
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Ensure you have Ollama installed and running on your system. Visit [Ollama's website](https://ollama.ai/) for installation instructions.

## Usage

1. Set up your environment variables:
   - Create a `.env` file in the root directory and add your OpenAI API key if you wish to use OpenAI models:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   - If you prefer using the Ollama model to avoid API costs, skip adding the API key.

2. Run the main script:
   ```bash
   streamlit run main.py
   ```
### Using the Streamlit Interface

- Fill in your personal details and fitness goals using the input fields.
- Click "Create Fitness Plan" to receive a personalized fitness plan generated by the AI.
- Provide feedback about your current plan in the "Update Fitness Plan" tab to get a revised plan.


## Project Structure

- `main.py`: The main script that sets up the AI Fitness Coach and streamlit UI.
- `requirements.txt`: List of Python package dependencies.

## Choosing Between OpenAI and Ollama

- **OpenAI Models**: Use these if you have an API key and don't mind incurring costs for high-quality language model responses.
- **Ollama Models**: A great cost-saving option that runs models locally on your system. Make sure to install Ollama and have it running.
  
## Contributing

Contributions to improve the AI Fitness Coach are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the LangChain and LangGraph frameworks.
- The Streamlit library is used for the user interface.
- Ollama is used for running the language models locally.
