import streamlit as st
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# State definition
class State(TypedDict):
    user_data: dict
    fitness_plan: str
    feedback: str
    progress: List[str]
    messages: Annotated[list, add_messages]

# Utility function to get OpenAI LLM
def get_openai_llm():
    return ChatOpenAI(api_key=openai_api_key,model="gpt-4o-mini", temperature=0)
# Utility function to get Ollama LLM
def get_ollama_llm(model_name="tinyllama"):
    print(f"Creating ChatOllama with model: {model_name}")
    return ChatOllama(model=model_name)

# User Input Agent
def user_input_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness coach assistant. Process the following user information:

        {user_input}

        Create a structured user profile based on this information. Include all relevant details for creating a personalized fitness plan.
        Return the profile as a valid JSON string."""
    )
    chain = prompt | llm | StrOutputParser()
    user_profile = chain.invoke({"user_input": json.dumps(state["user_data"])})
    try:
        state["user_data"] = json.loads(user_profile)
    except json.JSONDecodeError:
        pass
    state["messages"].append(AIMessage(content=f"Processed user profile: {json.dumps(state['user_data'], indent=2)}"))
    return state
# routine generation agent
def routine_generation_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness coach. Create a personalized fitness routine based on the following user data:

        {user_data}

        Create a detailed weekly fitness plan that includes:
        1. Types of exercises
        2. Duration and frequency of workouts
        3. Intensity levels
        4. Rest days
        5. Any dietary recommendations

        Present the plan in a clear, structured format."""
    )
    chain = prompt | llm | StrOutputParser()
    plan = chain.invoke({"user_data": json.dumps(state["user_data"])})
    state["fitness_plan"] = plan
    state["messages"].append(AIMessage(content=f"Generated fitness plan: {plan}"))
    return state

# Feedback Collection Agent
def feedback_collection_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness coach assistant. Analyze the following user feedback on their recent workout session:

        Current fitness plan: {current_plan}
        User feedback: {user_feedback}

        Summarize the user's feedback and suggest any immediate adjustments."""
    )
    chain = prompt | llm | StrOutputParser()
    feedback_summary = chain.invoke({"current_plan": state["fitness_plan"], "user_feedback": state["feedback"]})
    state["messages"].append(AIMessage(content=f"Feedback analysis: {feedback_summary}"))
    return state

# Routine Adjustment Agent
def routine_adjustment_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness coach. Adjust the current fitness plan based on the user's feedback:

        Current Plan:
        {current_plan}

        User Feedback:
        {feedback}

        Provide an updated weekly fitness plan that addresses the user's feedback while maintaining the overall structure and goals."""
    )
    chain = prompt | llm | StrOutputParser()
    updated_plan = chain.invoke({"current_plan": state["fitness_plan"], "feedback": state["feedback"]})
    state["fitness_plan"] = updated_plan
    state["messages"].append(AIMessage(content=f"Updated fitness plan: {updated_plan}"))
    return state

# Progress Monitoring Agent
def progress_monitoring_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI fitness progress tracker. Review the user's progress and provide encouragement or suggestions:

        User Data: {user_data}
        Current Plan: {current_plan}
        Progress History: {progress_history}

        Provide a summary of the user's progress, offer encouragement, and suggest any new challenges or adjustments."""
    )
    chain = prompt | llm | StrOutputParser()
    progress_update = chain.invoke(
        {"user_data": str(state["user_data"]), "current_plan": state["fitness_plan"], "progress_history": str(state["progress"])}
    )
    state["progress"].append(progress_update)
    state["messages"].append(AIMessage(content=f"Progress update: {progress_update}"))
    return state

# Motivational Agent
def motivational_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI motivational coach for fitness. Provide encouragement, tips, or reminders to the user:

        User Data: {user_data}
        Current Plan: {current_plan}
        Recent Progress: {recent_progress}

        Generate a motivational message, helpful tip, or reminder to keep the user engaged and committed to their fitness goals."""
    )
    chain = prompt | llm | StrOutputParser()
    motivation = chain.invoke(
        {"user_data": str(state["user_data"]), "current_plan": state["fitness_plan"], "recent_progress": state["progress"][-1] if state["progress"] else ""}
    )
    state["messages"].append(AIMessage(content=f"Motivation: {motivation}"))
    return state

# AIFitnessCoach class
class AIFitnessCoach:
    def __init__(self):
        print("Initializing AIFitnessCoach")
        self.llm = get_openai_llm()
        # self.llm = get_ollama_llm() you can uncomment this if you prefer to use the locally running llms
        self.graph = self.create_graph()

    def create_graph(self):
        print("Creating graph")
        workflow = StateGraph(State)

        # Define nodes
        workflow.add_node("user_input", lambda state: user_input_agent(state, self.llm))
        workflow.add_node("routine_generation", lambda state: routine_generation_agent(state, self.llm))
        workflow.add_node("feedback_collection", lambda state: feedback_collection_agent(state, self.llm))
        workflow.add_node("routine_adjustment", lambda state: routine_adjustment_agent(state, self.llm))
        workflow.add_node("progress_monitoring", lambda state: progress_monitoring_agent(state, self.llm))
        workflow.add_node("motivation", lambda state: motivational_agent(state, self.llm))

        # Define edges
        workflow.add_edge("user_input", "routine_generation")
        workflow.add_edge("routine_generation", "feedback_collection")
        workflow.add_edge("feedback_collection", "routine_adjustment")
        workflow.add_edge("routine_adjustment", "progress_monitoring")
        workflow.add_edge("progress_monitoring", "motivation")
        workflow.add_edge("motivation", END)

        # Set entry point
        workflow.set_entry_point("user_input")
        print("Graph created")
        return workflow.compile()

    def run(self, user_input):
        print("Running AIFitnessCoach")
        initial_state = State(
            user_data=user_input,
            fitness_plan="",
            feedback="",
            progress=[],
            messages=[HumanMessage(content=json.dumps(user_input))]
        )
        print(f"Initial state: {initial_state}")
        final_state = self.graph.invoke(initial_state)
        print(f"Final state: {final_state}")
        return final_state["messages"]

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Fitness Coach", layout="wide")
    st.title("AI Fitness Coach")

    # Initialize session state
    if "fitness_coach" not in st.session_state:
        st.session_state.fitness_coach = AIFitnessCoach()

    tab1, tab2 = st.tabs(["Create Fitness Plan", "Update Fitness Plan"])

    with tab1:
        st.header("Create Your Personalized Fitness Plan")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120)
        with col2:
            weight = st.number_input("Weight (kg)", min_value=1.0)
        with col3:
            height = st.number_input("Height (cm)", min_value=1.0)
        
        gender = st.radio("Gender", ["Male", "Female", "Other"])
        
        primary_goal = st.selectbox(
            "Primary Goal",
            ["Weight loss", "Muscle gain", "Endurance improvement", "General fitness"]
        )
        
        target_timeframe = st.selectbox(
            "Target Timeframe",
            ["3 months", "6 months", "1 year"]
        )
        
        workout_preferences = st.multiselect(
            "Workout Type Preferences",
            ["Cardio", "Strength training", "Yoga", "Pilates", "Flexibility exercises", "HIIT"]
        )
        
        workout_duration = st.slider(
            "Preferred Workout Duration (minutes)",
            min_value=15,
            max_value=120,
            step=15
        )
        
        workout_days = st.multiselect(
            "Preferred Workout Days",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        
        activity_level = st.radio(
            "Current Activity Level",
            ["Sedentary", "Lightly active", "Moderately active", "Highly active"]
        )
        
        health_conditions = st.text_area("Health Conditions or Injuries")
        dietary_preferences = st.text_area("Dietary Preferences (Optional)")
        
        if st.button("Create Fitness Plan"):
            user_data = {
                "age": age,
                "weight": weight,
                "height": height,
                "gender": gender,
                "primary_goal": primary_goal,
                "target_timeframe": target_timeframe,
                "workout_preferences": workout_preferences,
                "workout_duration": workout_duration,
                "workout_days": workout_days,
                "activity_level": activity_level,
                "health_conditions": health_conditions,
                "dietary_preferences": dietary_preferences
            }
            
            with st.spinner("Generating your personalized fitness plan..."):
                messages = st.session_state.fitness_coach.run(user_data)
                st.session_state.last_plan = messages
                
            for message in messages:
                st.write(f"**{message.type.capitalize()}:** {message.content}")

    with tab2:
        st.header("Update Your Fitness Plan")
        feedback = st.text_area("Provide feedback about your current plan:")
        
        if st.button("Update Plan"):
            with st.spinner("Updating your fitness plan..."):
                messages = st.session_state.fitness_coach.run({"feedback": feedback})
                
            for message in messages:
                st.write(f"**{message.type.capitalize()}:** {message.content}")

if __name__ == "__main__":
    main()
