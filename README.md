# 📝 Research Backed Blog Generation Platform
A personal AI-powered platform for generating structured long form content, featuring authenticated access, a user dashboard, and a multi-stage content generation pipeline with research integration.

---

## Overview

This project is a **full-stack content generation platform** that allows users to log in securely and generate high-quality blog drafts through an interactive dashboard.

Unlike basic text generators, the system combines:

* Structured planning
* Research-backed content generation
* Parallel execution
* Output aggregation

to produce coherent and contextually relevant blog content.

---

## Authentication

* Integrated **Google OAuth authentication**
* Restricts access to authorized users only
* Ensures secure, personalized usage of the platform

---

## Key Features

* **User Dashboard**
  Generate, view, and manage blog content in a single interface

* **Multi-Stage Content Pipeline**
  Structured workflow for planning, generation, and refinement

* **Dynamic Routing**
  Automatically switches between direct generation and research-backed workflows

* **Research Integration**
  Retrieves external information to improve factual grounding

* **Parallel Processing**
  Executes subtasks concurrently for faster generation

* **Content Aggregation**
  Combines outputs into a structured, readable blog

---

## System Architecture

```
 User Login (Google OAuth)
        ↓
 Dashboard UI
        ↓
 Topic Input
        ↓
 Planner Module
        ↓
 Routing Layer ───────────────┐
        ↓                     ↓
 Direct Generation    Research Pipeline
        |                     ↓
        |             External Retrieval
        |                     ↓
        |             Content Generation
        ↓                     ↓
        Parallel Workers (Subtasks)
                    ↓
            Aggregation Layer
                    ↓
            Final Blog Output(.md/.html(predefined template))
```

---

## Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **Frameworks:** LangGraph, LangChain
* **Authentication:** Google OAuth
* **Retrieval APIs:** Tavily, Arxiv
* **LLMs:** OpenAI / Groq

---

## How It Works

1. **User Authentication**
   Users log in via Google OAuth

2. **Dashboard Interaction**
   Users input a topic through the UI

3. **Planning**
   The system decomposes the topic into structured sections

4. **Routing Decision**
   Determines whether each section needs research

5. **Execution**

   * Direct generation for simple content
   * Retrieval + generation for complex topics

6. **Parallel Processing**
   Multiple sections processed simultaneously

7. **Aggregation**
   Outputs combined into a complete blog

---

## Installation

```bash
git clone https://github.com/ashwinibhardwaj/blog-writing-agent.git
cd blog-writing-agent
pip install -r requirements.txt
```

---

## Usage

```bash
python app.py
```

* Login via Google
* Enter a topic in the dashboard
* Generate structured blog content

## Contact

**Ashwini Bhardwaj**

* GitHub: https://github.com/ashwinibhardwaj
* LinkedIn: https://www.linkedin.com/in/ashwini-bhardwaj/

---

## ⭐ If you found this useful, consider giving it a star!
