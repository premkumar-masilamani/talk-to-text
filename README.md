# Coaching Insights

This project provides a robust solution for transcribing coaching audio files and generating AI-powered feedback based on predefined prompt levels. It leverages `whisper.cpp` for efficient audio transcription and allows for flexible integration with various AI models for deeper analysis.

## Features

-   **High-Quality Audio Transcription**: Utilizes `whisper.cpp` for accurate and fast transcription of audio files.
-   **Automated Workflow**: Integrates with a `Makefile` for streamlined transcription processes.
-   **AI-Powered Feedback**: Facilitates the generation of detailed coaching feedback using custom AI prompts.
-   **Tiered Feedback Levels**: Supports different coaching levels (ACC, PCC, MCC) with specific prompt JSON files.
-   **Client-Centric Insights**: Provides a dedicated prompt for generating actionable insights for clients.

## Pre-Requisites

Before you begin, ensure you have the following installed:

-   **Python 3.8+**: The project is developed with Python.
-   **pipenv**: For managing project dependencies. If you don't have it, install with `pip install pipenv`.
-   **Make**: A build automation tool.
-   **FFmpeg**: Essential for audio processing. The `Makefile` typically handles the download and setup of `ffmpeg` if it's not found, but you might need to install it manually on some systems.

## Setup

Simply run the following command to set up your environment:

```bash
make install
```

This command will:
- Install Python dependencies using `pipenv`.

## Usage

Once the setup is complete, you can run the entire transcription and feedback generation pipeline with a single command:

```bash
make run
```

This command will:
- Expect your coaching audio files to be placed in the `files/` directory.
- Transcribe all audio files in `files/` using `whisper.cpp`.
- Generate transcript files in the same `files/` directory.

**PS**: The code will try to download the AI model for the first time you run it. It may take a while depending on your internet connection.

## AI Feedback

You can use the generated transcript and the appropriate prompt from the `prompts/` directory with your chosen AI model to get detailed feedback. Simply upload the transcript file and prompt file to your AI model's website. Please explore different AI models (ChatGPT, Gemini, Grok etc) and use the feedback that matches your needs.

The `prompts/` directory contains several JSON files, each designed to elicit specific types of feedback from an AI:

-   `01-acc-coach-prompt.json`: Focuses on foundational coaching competencies as per ACC standards, providing a baseline evaluation covering core ICF competencies and ethical practice.
-   `02-pcc-coach-prompt.json`: Aims for more advanced coaching skills and nuanced feedback aligned with PCC standards. This prompt builds upon the ACC evaluation by including additional sections for NLP Neurological Levels Mapping and Belief & Identity Structure Work, requiring a deeper analysis of the coach's techniques.
-   `03-mcc-coach-prompt.json`: For expert-level coaching analysis, adhering to MCC criteria. This is the most comprehensive evaluation, incorporating all elements of the PCC prompt and adding a section for "MCC-Level Distinctions & Hallmarks," which assesses the subtle and masterful qualities of highly experienced coaching.
-   `client-prompt.json`: Extracts key takeaways, action items, and growth areas directly relevant to the client, providing them with actionable insights.

Each JSON prompt file contains detailed instructions that guide the AI on the structure, tone, and focus of the feedback it should generate. The progression from ACC to PCC to MCC prompts reflects an increasing expectation in the depth, nuance, and theoretical frameworks applied to the coaching evaluation.
