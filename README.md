# Story Generator

This is a story generation system using LangGraph to orchestrate a set of agents that collaborate to generate a story from a user prompt.

## Overview

The system consists of three phases:

1. **Pre-Writing & Architectural Design**: Establishes the foundation of the story
   - Story Architect Agent: Defines the core premise, genre, themes, and plot structure
   - World-Building Agent: Creates the setting, history, geography, and unique systems
   - Character Development Agent: Designs detailed characters with backstories and motivations
   - Outline Generator Agent: Expands the concepts into a detailed scene-by-scene outline

2. **Drafting & Iterative Feedback**: Generates and refines the story chapter by chapter
   - Narrative Agent: Generates the actual prose for chapters or scenes
   - Pacing Agent: Analyzes the flow and pacing of the text
   - Continuity Agent: Checks for inconsistencies in plot, character details, or world-building
   - Emotional Resonance Agent: Evaluates the emotional impact of scenes
   - Dialogue Polisher Agent: Refines dialogue to sound natural and purposeful
   - Sensory Detail Agent: Adds vivid descriptions that appeal to the five senses

3. **Final Polishing & Delivery**: Finalizes the manuscript and provides reader feedback
   - Refinement Editing Agent (under construction): Performs line-by-line edits for grammar, syntax, and readability
   - Reader Simulation Agent (not implemented): Simulates a reader's experience and provides feedback

## Requirements

- Python 3.8+
- LangGraph library
- Access to an LLM API (default: LMStudio running locally)

## Configuration

The system is configured to use LMStudio running locally at http://localhost:1234/v1. You can modify the configuration in the `Config` class:

```python
class Config:
    # LLM endpoint config
    LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
    LM_MODEL = os.getenv("LM_MODEL", "qwen2.5-coder-7b-instruct")
```

## Usage

1. Ensure you have an LLM service running (e.g., LMStudio)
2. Run the script:

```bash
python story_generator.py
```

3. Enter a story idea or prompt when prompted
4. The system will generate a story based on your prompt, using multiple agents to collaborate on different aspects of the story

## Example

```
=== Story Generation System ===
Enter a story idea or prompt: A detective with the ability to speak to ghosts investigates a murder in a small coastal town.

Generating a story based on: 'A detective with the ability to speak to ghosts investigates a murder in a small coastal town.'

This process will take some time as multiple agents collaborate to create your story...

=== Your Generated Story ===

[The generated story will appear here]

=== Reader Review ===

[A simulated reader's review will appear here]
```

## Implementation Details

The system uses LangGraph to orchestrate the agents in a workflow. The workflow is defined in the `StoryGraphBuilder` class, which creates a graph with nodes for each agent and edges to connect them.

The main components are:

- `StoryState`: A TypedDict that represents the state of the story generation process
- Agent classes: Each agent is responsible for a specific aspect of the story generation process
- `StoryGraphBuilder`: Builds the graph that connects the agents
- `StoryGenerator`: The main class that creates the agents, builds the graph, and runs the story generation process

The graph is structured to follow the three phases of story generation, with a cyclical feedback loop in the drafting phase to iteratively improve each chapter.

## Extending the System

You can extend the system by:

1. Adding new agent types for specialized tasks
2. Modifying the prompts for existing agents
3. Changing the graph structure to alter the workflow
4. Adding more feedback loops or conditional paths

## Limitations

- The system relies on an external LLM service
- The quality of the generated story depends on the capabilities of the underlying LLM
- The process can be time-consuming, especially for longer stories
- There is limited user interaction during the generation process