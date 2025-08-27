"""
Story generation system using LangGraph with Template Method pattern.

"""

import os
import json
from typing import List, Dict, Any, TypedDict, Optional, Literal, Union, Type
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from operator import add
from abc import ABC, abstractmethod

# Configuration constants
class Config:
    LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
    LM_MODEL = os.getenv("LM_MODEL", "gemma-3-12b-it") # "qwen2.5-coder-7b-instruct")

# Pydantic models for agent outputs
class StorySummaryStructure(BaseModel):
    act1: str = Field(default="Setup", description="First act description")
    inciting_incident: str = Field(default="Not specified", description="Inciting incident description")
    act2: str = Field(default="Confrontation", description="Second act description")
    midpoint: str = Field(default="Not specified", description="Midpoint description")
    act3: str = Field(default="Resolution", description="Third act description")
    climax: str = Field(default="Not specified", description="Climax description")

class StorySummaryOutput(BaseModel):
    premise: Optional[str] = Field(default=None, description="Story premise")
    genre: str = Field(default="Not specified", description="Genre of the story")
    target_audience: str = Field(default="General", description="Intended audience")
    structure: StorySummaryStructure = Field(default_factory=StorySummaryStructure, description="Narrative structure details")
    central_conflict: str = Field(default="Not specified", description="Central conflict")
    themes: List[str] = Field(default=["Not specified"], description="List of themes")

class WorldGuideOutput(BaseModel):
    history: str = Field(default="Not specified", description="History of the world/setting")
    geography: str = Field(default="Not specified", description="Geography and physical environment")
    culture: str = Field(default="Not specified", description="Key social rules and cultural elements")
    unique_systems: str = Field(default="Not specified", description="Unique systems (e.g., magic, technology)")

class CharacterProfile(BaseModel):
    name: str = Field(description="Character name")
    personality: str = Field(default="Not specified", description="Core personality traits")
    backstory: str = Field(default="Not specified", description="Character backstory")
    motivations: Dict[str, str] = Field(default={"conscious": "Not specified", "unconscious": "Not specified"}, description="Conscious and unconscious motivations")
    relationships: List[str] = Field(default=[], description="Key relationships")
    character_arc: str = Field(default="Not specified", description="Character arc")

class CharacterProfiles(BaseModel):
    character_profiles: List[CharacterProfile] = Field(default=[], description="List of characters profiles")

class StoryOutlineOutput(BaseModel):
    outline: List[Dict[str, Any]] = Field(default=[{"chapter": 1, "title": "Beginning", "purpose": "Introduce the protagonist and the world", "characters": ["Protagonist"], "summary": "The story begins..."}], description="Scene-by-scene or chapter-by-chapter outline")

class PacingFeedback(BaseModel):
    pacing_feedback: str = Field(default="Not specified", description="Feedback on pacing issues")
    issues: List[str] = Field(default=[], description="List of pacing issues")

class ContinuityFeedback(BaseModel):
    continuity_feedback: str = Field(default="Not specified", description="Feedback on continuity issues")
    inconsistencies: List[str] = Field(default=[], description="List of continuity inconsistencies")

class EmotionalFeedback(BaseModel):
    emotional_feedback: str = Field(default="Not specified", description="Feedback on emotional impact")
    flat_moments: List[str] = Field(default=[], description="Moments that feel emotionally flat")
    suggestions: List[str] = Field(default=[], description="Suggestions to deepen emotional connection")

# State definition
class StoryState(TypedDict, total=False):
    user_prompt: str
    story_summary: Dict[str, Any]
    world_guide: Dict[str, Any]
    character_profiles: List[Dict[str, Any]]
    story_outline: Dict[str, Any]
    current_chapter_index: int
    current_chapter_draft: str
    feedback: Dict[str, Any]
    polished_draft: str
    manuscript: Annotated[List[str], add]
    final_manuscript: str
    processing_stage: str

# BaseAgent with Template Method
class BaseAgent(ABC):
    def __init__(self, llm_client: ChatOpenAI):
        self.llm_client = llm_client

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_output_model(self) -> Type[BaseModel]:
        pass

    @abstractmethod
    def get_input_data(self, state: StoryState) -> Any:
        pass

    @abstractmethod
    def get_state_update_key(self) -> str:
        pass

    @abstractmethod
    def get_processing_stage(self) -> str:
        pass

    def __call__(self, input_data: Any) -> Union[Dict[str, Any], str]:
        print(f"\n[{self.__class__.__name__}] Starting processing...")
        top_system_prompt = """
        You are a helpful story generating assistant that will return the expected responses without adding self-dialogue about the story or commands being received to help write the story.
        """
        messages = [
            SystemMessage(content=top_system_prompt),
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"Input: {input_data if isinstance(input_data, str) else json.dumps(input_data, indent=2)}")
        ]
        try:
            if self.get_output_model() == str:
                resp = self.llm_client.invoke(messages, temperature=0.7, max_tokens=2000)
                return resp.content
            else:
                chat = self.llm_client.with_structured_output(self.get_output_model())
                resp = chat.invoke(messages, temperature=0.7)
                return resp.model_dump()
        except Exception as e:
            print(f"Error generating output: {e}")
            if self.get_output_model() == str:
                return "Not specified"
            default_instance = self.get_output_model()()
            return default_instance.model_dump()

    def node_function(self, state: StoryState) -> Dict[str, Any]:
        print(f"\n[{self.__class__.__name__} Node] Processing state...")
        input_data = self.get_input_data(state)
        output = self(input_data)
        return {
            self.get_state_update_key(): output,
            "processing_stage": self.get_processing_stage()
        }

# Agent Implementations
class StoryArchitectAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Story Architect Agent. Your role is to define the foundation of a story.

        Based on the core idea provided, you will:
        1. Define the main premise, genre, and target audience
        2. Outline the three-act structure, identifying the inciting incident, midpoint, and climax
        3. Determine the central conflict and the major themes the story will explore

        The output should match the StorySummaryOutput schema.
        """

    def get_output_model(self) -> Type[BaseModel]:
        return StorySummaryOutput

    def get_input_data(self, state: StoryState) -> Any:
        return state["user_prompt"]

    def get_state_update_key(self) -> str:
        return "story_summary"

    def get_processing_stage(self) -> str:
        return "story_architecture_complete"

class WorldBuildingAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the World-Building Agent. Your role is to design the story's setting.

        Given the plot and themes from the Story Architect, you will detail:
        1. History of the world/setting
        2. Geography and physical environment
        3. Key social rules and cultural elements
        4. Unique systems (e.g., magic, technology, political structure)

        The output should match the WorldGuideOutput schema.
        """

    def get_output_model(self) -> Type[BaseModel]:
        return WorldGuideOutput

    def get_input_data(self, state: StoryState) -> Any:
        return state["story_summary"]

    def get_state_update_key(self) -> str:
        return "world_guide"

    def get_processing_stage(self) -> str:
        return "world_building_complete"

class CharacterDevelopmentAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Character Development Agent. Your role is to create detailed profiles for the main characters.

        Using the plot and themes from the Story Architect, and the setting from the World-Building Agent, 
        for each character, include:
        1. Core personality traits
        2. Backstory
        3. Motivations (both conscious and unconscious)
        4. Key relationships
        5. Character arc

        The output should be a list of CharacterProfile schemas.
        """

    def get_output_model(self) -> Type[BaseModel]:
        return CharacterProfiles

    def get_input_data(self, state: StoryState) -> Any:
        return {"story_summary": state["story_summary"], "world_guide": state["world_guide"]}

    def get_state_update_key(self) -> str:
        return "character_profiles"

    def get_processing_stage(self) -> str:
        return "character_development_complete"

    def __call__(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = super().__call__(input_data)
        # Ensure output is a list of profiles
        return [result] if isinstance(result, dict) else result

class OutlineGeneratorAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Outline Generator Agent. Your role is to create a comprehensive, scene-by-scene outline.

        Synthesize the outputs from the Story Architect, World-Building, and Character Development agents to create an outline where each scene has:
        1. A brief description of its purpose
        2. The characters involved
        3. A summary of the action and dialogue

        The output should match the StoryOutlineOutput schema.
        """

    def get_output_model(self) -> Type[BaseModel]:
        return StoryOutlineOutput

    def get_input_data(self, state: StoryState) -> Any:
        return {
            "story_summary": state["story_summary"],
            "world_guide": state["world_guide"],
            "character_profiles": state["character_profiles"]
        }

    def get_state_update_key(self) -> str:
        return "story_outline"

    def get_processing_stage(self) -> str:
        return "outline_complete"

class NarrativeAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Narrative Agent. Your role is to write the next chapter/scene of the story.

        Using the outline as a guide, focus on:
        1. Developing the prose
        2. Crafting dialogue that feels natural
        3. Describing the action as outlined

        Ensure the chapter ends with a hook. Return the prose as a string.
        """

    def get_output_model(self) -> Union[Type[BaseModel], Type[str]]:
        return str

    def get_input_data(self, state: StoryState) -> Any:
        story_outline = state.get("story_outline", {}).get("outline", [])
        current_chapter_index = state.get("current_chapter_index", 0)
        feedback = state.get("feedback", {})
        return {"outline_segment": story_outline[current_chapter_index] if current_chapter_index < len(story_outline) else {}, "feedback": feedback}

    def get_state_update_key(self) -> str:
        return "current_chapter_draft"

    def get_processing_stage(self) -> str:
        return "draft_complete"

class PacingAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Pacing Agent. Analyze the draft for pacing issues and provide specific feedback.
        Focus on sections where the pace is too fast or too slow.
        The output should match the PacingFeedback schema.
        """

    def get_output_model(self) -> Type[BaseModel]:
        return PacingFeedback

    def get_input_data(self, state: StoryState) -> Any:
        return state["current_chapter_draft"]

    def get_state_update_key(self) -> str:
        return "pacing_feedback"

    def get_processing_stage(self) -> str:
        return "pacing_feedback_complete"

class ContinuityAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Continuity Agent. Analyze the provided chapter against the story's overall outline, world-building guide, and character profiles.

        Point out any inconsistencies in:
        1. Plot
        2. Character behavior
        3. World lore

        The output should match the ContinuityFeedback schema.
        """

    def get_output_model(self) -> Type[BaseModel]:
        return ContinuityFeedback

    def get_input_data(self, state: StoryState) -> Any:
        return {
            "draft": state["current_chapter_draft"],
            "story_outline": state["story_outline"],
            "world_guide": state["world_guide"],
            "character_profiles": state["character_profiles"]
        }

    def get_state_update_key(self) -> str:
        return "continuity_feedback"

    def get_processing_stage(self) -> str:
        return "continuity_feedback_complete"

class EmotionalResonanceAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Emotional Resonance Agent. Evaluate the chapter for emotional impact.

        Identify:
        1. Moments that feel emotionally flat or unearned
        2. Suggestions to deepen the emotional connection

        The output should match the EmotionalFeedback schema.
        """

    def get_output_model(self) -> Type[BaseModel]:
        return EmotionalFeedback

    def get_input_data(self, state: StoryState) -> Any:
        return state["current_chapter_draft"]

    def get_state_update_key(self) -> str:
        return "emotional_feedback"

    def get_processing_stage(self) -> str:
        return "emotional_feedback_complete"

class DialoguePolisherAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Dialogue Polisher Agent. Edit the dialogue in the provided chapter.

        Ensure each character's voice is:
        1. Distinct and consistent with their profile
        2. Natural and dynamic
        3. Serving a purpose

        Return the polished chapter as a string.
        """

    def get_output_model(self) -> Union[Type[BaseModel], Type[str]]:
        return str

    def get_input_data(self, state: StoryState) -> Any:
        return {"draft": state["current_chapter_draft"], "character_profiles": state["character_profiles"]}

    def get_state_update_key(self) -> str:
        return "polished_draft"

    def get_processing_stage(self) -> str:
        return "dialogue_polished"

class SensoryDetailAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Sensory Detail Agent. Enhance the chapter by adding sensory details (sight, sound, smell, touch, taste).

        Identify opportunities to:
        1. Replace general descriptions with vivid imagery
        2. Add details that immerse the reader
        3. Engage all five senses where appropriate

        Return the enhanced chapter as a string.
        """

    def get_output_model(self) -> Union[Type[BaseModel], Type[str]]:
        return str

    def get_input_data(self, state: StoryState) -> Any:
        return state["polished_draft"]

    def get_state_update_key(self) -> str:
        return "polished_draft"

    def get_processing_stage(self) -> str:
        return "sensory_details_added"

class RefinementEditingAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return """
        You are the Refinement/Editing Agent. Perform a final, line-by-line edit on the manuscript.

        Correct:
        1. Grammar, punctuation, and spelling errors
        2. Improve syntax and word choice
        3. Ensure a clear, concise, professional tone

        Return the final manuscript as a string.
        """

    def get_output_model(self) -> Union[Type[BaseModel], Type[str]]:
        return str

    def get_input_data(self, state: StoryState) -> Any:
        return "\n\n".join(state["manuscript"])

    def get_state_update_key(self) -> str:
        return "final_manuscript"

    def get_processing_stage(self) -> str:
        return "manuscript_finalized"

# StoryGraphBuilder
class StoryGraphBuilder:
    def __init__(self, *agents):
        self.agents = {
            "story_architect": agents[0],
            "world_building": agents[1],
            "character_development": agents[2],
            "outline_generator": agents[3],
            "narrative": agents[4],
            "pacing": agents[5],
            "continuity": agents[6],
            "emotional_resonance": agents[7],
            "dialogue_polisher": agents[8],
            "sensory_detail": agents[9],
            "refinement_editing": agents[10]
        }

    def collect_feedback_node(self, state: StoryState) -> Dict[str, Any]:
        print("\n[FeedbackNode] Collecting comprehensive feedback...")
        feedback = {
            "pacing": self.agents["pacing"].node_function(state)["pacing_feedback"],
            "continuity": self.agents["continuity"].node_function(state)["continuity_feedback"],
            "emotional": self.agents["emotional_resonance"].node_function(state)["emotional_feedback"],
            "timestamp": "collected"
        }
        return {
            "feedback": feedback,
            "processing_stage": "feedback_collected"
        }

    def revise_chapter_node(self, state: StoryState) -> Dict[str, Any]:
        print("\n[ReviseNode] Revising chapter based on feedback...")
        revised_draft = self.agents["narrative"](self.agents["narrative"].get_input_data(state))
        return {
            "current_chapter_draft": revised_draft,
            "processing_stage": "chapter_revised"
        }

    def polish_chapter_node(self, state: StoryState) -> Dict[str, Any]:
        print("\n[PolishNode] Polishing chapter comprehensively...")
        draft = state.get("current_chapter_draft", "")
        polished_draft = self.agents["dialogue_polisher"]({"draft": draft, "character_profiles": state["character_profiles"]})
        polished_draft = self.agents["sensory_detail"](polished_draft)
        return {
            "polished_draft": polished_draft,
            "processing_stage": "chapter_polished"
        }

    def add_to_manuscript_node(self, state: StoryState) -> Dict[str, Any]:
        print("\n[AddToManuscriptNode] Adding chapter to manuscript...")
        polished_draft = state.get("polished_draft", "")
        current_chapter_index = state.get("current_chapter_index", 0)
        if polished_draft:
            os.makedirs("draft_output", exist_ok=True)
            chapter_path = os.path.join("draft_output", f"chapter_{current_chapter_index + 1}.txt")
            with open(chapter_path, "w", encoding="utf-8") as f:
                f.write(polished_draft)
            print(f"[Checkpoint] Saved Chapter {current_chapter_index + 1} to {chapter_path}")
        return {
            "manuscript": [polished_draft],
            "current_chapter_index": current_chapter_index + 1,
            "current_chapter_draft": "",
            "polished_draft": "",
            "feedback": {},
            "processing_stage": "chapter_added_to_manuscript"
        }

    def finalize_manuscript_node(self, state: StoryState) -> Dict[str, Any]:
        print("\n[FinalizeNode] Finalizing manuscript...")
        final_manuscript = self.agents["refinement_editing"](self.agents["refinement_editing"].get_input_data(state))
        return {
            "final_manuscript": final_manuscript,
            "processing_stage": "manuscript_finalized"
        }

    def should_revise_or_polish(self, state: StoryState) -> Literal["revise", "polish"]:
        feedback = state.get("feedback", {})
        processing_stage = state.get("processing_stage", "")
        continuity_feedback = feedback.get("continuity", {})
        inconsistencies = continuity_feedback.get("inconsistencies", [])
        if processing_stage == "feedback_collected" and len(inconsistencies) > 0:
            print("[FeedbackRouter] Found continuity issues, revising chapter...")
            return "revise"
        print("[FeedbackRouter] Feedback looks good, proceeding to polish...")
        return "polish"

    def should_continue_drafting(self, state: StoryState) -> Literal["continue_drafting", "finalize"]:
        current_chapter_index = state["current_chapter_index"]
        story_outline = state.get("story_outline", {}).get("outline", [])
        print(f"[Router] Chapter {current_chapter_index + 1} of {len(story_outline)}")
        if current_chapter_index >= len(story_outline):
            print("[Router] All chapters complete, finalizing...")
            return "finalize"
        print("[Router] Continuing with next chapter...")
        return "continue_drafting"

    def should_start_drafting_or_finalize(self, state: StoryState) -> Literal["start_drafting", "finalize"]:
        story_outline = state.get("story_outline", {}).get("outline", [])
        if not story_outline:
            print("[Router] Outline is empty; skipping drafting and finalizing.")
            return "finalize"
        print("[Router] Outline present; starting drafting.")
        return "start_drafting"

    def should_collect_feedback_or_finalize(self, state: StoryState) -> Literal["collect_feedback", "finalize"]:
        story_outline = state.get("story_outline", {}).get("outline", [])
        current_chapter_index = state.get("current_chapter_index", 0)
        if current_chapter_index >= len(story_outline):
            print("[Router] No chapter to draft; finalizing.")
            return "finalize"
        if not state.get("current_chapter_draft"):
            print("[Router] Draft missing; finalizing to avoid dead-end.")
            return "finalize"
        return "collect_feedback"

    def build(self):
        workflow = StateGraph(StoryState)
        workflow.add_node("story_architect", self.agents["story_architect"].node_function)
        workflow.add_node("world_building", self.agents["world_building"].node_function)
        workflow.add_node("character_development", self.agents["character_development"].node_function)
        workflow.add_node("outline_generator", self.agents["outline_generator"].node_function)
        workflow.add_node("narrative", self.agents["narrative"].node_function)
        workflow.add_node("feedback", self.collect_feedback_node)
        workflow.add_node("revise", self.revise_chapter_node)
        workflow.add_node("polish", self.polish_chapter_node)
        workflow.add_node("add_to_manuscript", self.add_to_manuscript_node)
        workflow.add_node("finalize", self.finalize_manuscript_node)

        workflow.add_edge(START, "story_architect")
        workflow.add_edge("story_architect", "world_building")
        workflow.add_edge("world_building", "character_development")
        workflow.add_edge("character_development", "outline_generator")
        workflow.add_conditional_edges(
            "outline_generator",
            self.should_start_drafting_or_finalize,
            {"start_drafting": "narrative", "finalize": "finalize"}
        )
        workflow.add_conditional_edges(
            "narrative",
            self.should_collect_feedback_or_finalize,
            {"collect_feedback": "feedback", "finalize": "finalize"}
        )
        workflow.add_conditional_edges(
            "feedback",
            self.should_revise_or_polish,
            {"revise": "revise", "polish": "polish"}
        )
        workflow.add_edge("revise", "feedback")
        workflow.add_edge("polish", "add_to_manuscript")
        workflow.add_conditional_edges(
            "add_to_manuscript",
            self.should_continue_drafting,
            {"continue_drafting": "narrative", "finalize": "finalize"}
        )
        workflow.add_edge("finalize", END)
        return workflow.compile()

# StoryGenerator
class StoryGenerator:
    def __init__(self):
        self.llm_client = ChatOpenAI(
            base_url=Config.LMSTUDIO_URL,
            model=Config.LM_MODEL,
            api_key="not-needed",
            temperature=0.7,
            max_tokens=2000
        )
        self.agents = [
            StoryArchitectAgent(self.llm_client),
            WorldBuildingAgent(self.llm_client),
            CharacterDevelopmentAgent(self.llm_client),
            OutlineGeneratorAgent(self.llm_client),
            NarrativeAgent(self.llm_client),
            PacingAgent(self.llm_client),
            ContinuityAgent(self.llm_client),
            EmotionalResonanceAgent(self.llm_client),
            DialoguePolisherAgent(self.llm_client),
            SensoryDetailAgent(self.llm_client),
            RefinementEditingAgent(self.llm_client)
        ]
        self.graph_builder = StoryGraphBuilder(*self.agents)
        self.graph = self.graph_builder.build()

    def setup(self):
        try:
            import requests
            r = requests.get(Config.LMSTUDIO_URL.rstrip("/") + "/models", timeout=3)
            print("LLM service reachable at", Config.LMSTUDIO_URL)
        except Exception as e:
            print("Warning: LLM service not reachable at", Config.LMSTUDIO_URL)

    def run(self, user_prompt: str):
        print("=== Story Generation System ===")
        print(f"Generating story: '{user_prompt}'")
        init_state = {
            "user_prompt": user_prompt,
            "manuscript": [],
            "processing_stage": "starting"
        }
        initial_state = StoryState(**init_state)
        try:
            result = self.graph.invoke(initial_state, {"recursion_limit": 100})
            if "final_manuscript" in result:
                print("\n=== Generated Story ===\n")
                print(result["final_manuscript"])
                return result["final_manuscript"]
            else:
                print("Story generation incomplete.")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None

def main():
    generator = StoryGenerator()
    generator.setup()
    user_prompt = input("Enter story prompt: ").strip() or "A detective with the ability to speak to ghosts investigates a murder in a small coastal town."
    generator.run(user_prompt)

if __name__ == "__main__":
    main()