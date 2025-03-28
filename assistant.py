import sys
import json
import wave
import time
import pyttsx3
import torch
import requests
import soundfile
import yaml
import pygame
import pygame.locals
import numpy as np
import pyaudio
import whisper
import logging
import threading
import queue
import os
import subprocess
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Pygame and UI Constants
BACK_COLOR = (0,0,0)
REC_COLOR = (255,0,0)
TEXT_COLOR = (255,255,255)
REC_SIZE = 80
FONT_SIZE = 24
WIDTH = 320
HEIGHT = 240
KWIDTH = 20
KHEIGHT = 6
MAX_TEXT_LEN_DISPLAY = 32

# Audio Input Constants
INPUT_DEFAULT_DURATION_SECONDS = 5
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024
OLLAMA_REST_HEADERS = {'Content-Type': 'application/json'}
INPUT_CONFIG_PATH = "assistant.yaml"

class IntentClassifier:
    def __init__(self, ollama_url, ollama_model):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def classify_intent(self, text):
        """
        Classify the intent of the user's input
        Possible intents: 'application', 'file', 'search', 'general_query'
        """
        prompt = f"""Classify the intent of the following text into one of these categories:
        - application: Requests to open a specific application or program
        - file: Requests to list or open files/directories
        - search: Requests to search the internet or perform a web search
        - general_query: Any other type of query or conversation

        Text: "{text}"

        Respond ONLY with the intent category (application/file/search/general_query):"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                headers={'Content-Type': 'application/json'}
            )
            intent = response.json().get('response', '').strip().lower()
            
            # Fallback to basic intent detection if AI classification fails
            if intent not in ['application', 'file', 'search', 'general_query']:
                return self._fallback_intent_detection(text)
            
            return intent
        except Exception as e:
            logging.error(f"Intent classification error: {e}")
            return self._fallback_intent_detection(text)

    def _fallback_intent_detection(self, text):
        """
        Fallback method for intent detection using keyword matching
        """
        text = text.lower()
        
        # Application keywords
        app_keywords = ['open', 'launch', 'start', 'run']
        app_list = ['google', 'chrome', 'firefox', 'terminal', 'files', 'settings', 'calendar']
        
        # File keywords
        file_keywords = ['list files', 'show files', 'directory', 'folder']
        
        # Search keywords
        search_keywords = ['search', 'find', 'look up', 'google', 'internet']
        
        # Check for application intent
        if any(keyword in text for keyword in app_keywords) and \
           any(app in text for app in app_list):
            return 'application'
        
        # Check for file intent
        if any(keyword in text for keyword in file_keywords):
            return 'file'
        
        # Check for search intent
        if any(keyword in text for keyword in search_keywords):
            return 'search'
        
        # Default to general query
        return 'general_query'

class Assistant:
    def __init__(self):
        logging.info("Initializing Assistant")
        self.config = self.init_config()

        # Initialize intent classifier
        self.intent_classifier = IntentClassifier(
            self.config.ollama.url, 
            self.config.ollama.model
        )

        # Pygame initialization
        pygame.init()
        programIcon = pygame.image.load('assistant.png')

        self.clock = pygame.time.Clock()
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("Ubuntu Assistant")
        self.windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.audio = pyaudio.PyAudio()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 800)  
        self.tts.setProperty('volume', 0.9) 

        try:
            self.audio.open(format=INPUT_FORMAT,
                            channels=INPUT_CHANNELS,
                            rate=INPUT_RATE,
                            input=True,
                            frames_per_buffer=INPUT_CHUNK).close()
        except Exception as e:
            logging.error(f"Error opening audio stream: {str(e)}")
            self.wait_exit()

        self.display_message(self.config.messages.loadingModel)
        self.model = whisper.load_model("base.en")
        self.context = []

        self.text_to_speech(self.config.conversation.greeting)
        time.sleep(0.5)
        self.display_message(self.config.messages.pressSpace)

    def wait_exit(self):
        while True:
            self.display_message(self.config.messages.noAudioInput)
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    self.shutdown()

    def shutdown(self):
        logging.info("Shutting down Assistant")
        self.audio.terminate()
        pygame.quit()
        sys.exit()

    def init_config(self):
        logging.info("Initializing configuration")
        class Inst:
            pass

        with open('assistant.yaml', encoding='utf-8') as data:
            configYaml = yaml.safe_load(data)

        config = Inst()
        config.messages = Inst()
        config.messages.loadingModel = configYaml["messages"]["loadingModel"]
        config.messages.pressSpace = configYaml["messages"]["pressSpace"]
        config.messages.noAudioInput = configYaml["messages"]["noAudioInput"]

        config.conversation = Inst()
        config.conversation.greeting = configYaml["conversation"]["greeting"]

        config.ollama = Inst()
        config.ollama.url = configYaml["ollama"]["url"]
        config.ollama.model = configYaml["ollama"]["model"]

        config.whisperRecognition = Inst()
        config.whisperRecognition.modelPath = configYaml["whisperRecognition"]["modelPath"]
        config.whisperRecognition.lang = configYaml["whisperRecognition"]["lang"]

        return config

    def display_rec_start(self):
        logging.info("Displaying recording start")
        self.windowSurface.fill(BACK_COLOR)
        pygame.draw.circle(self.windowSurface, REC_COLOR, (WIDTH/2, HEIGHT/2), REC_SIZE)
        pygame.display.flip()

    def display_sound_energy(self, energy):
        logging.info(f"Displaying sound energy: {energy}")
        COL_COUNT = 5
        RED_CENTER = 100
        FACTOR = 10
        MAX_AMPLITUDE = 100

        self.windowSurface.fill(BACK_COLOR)
        amplitude = int(MAX_AMPLITUDE*energy)
        hspace, vspace = 2*KWIDTH, int(KHEIGHT/2)
        def rect_coords(x, y):
            return (int(x-KWIDTH/2), int(y-KHEIGHT/2),
                    KWIDTH, KHEIGHT)
        for i in range(-int(np.floor(COL_COUNT/2)), int(np.ceil(COL_COUNT/2))):
            x, y, count = WIDTH/2+(i*hspace), HEIGHT/2, amplitude-2*abs(i)

            mid = int(np.ceil(count/2))
            for i in range(0, mid):
                offset = i*(KHEIGHT+vspace)
                pygame.draw.rect(self.windowSurface, RED_CENTER,
                                rect_coords(x, y+offset))
                #mirror:
                pygame.draw.rect(self.windowSurface, RED_CENTER,
                                rect_coords(x, y-offset))
        pygame.display.flip()

    def display_message(self, text):
        logging.info(f"Displaying message: {text}")
        self.windowSurface.fill(BACK_COLOR)

        label = self.font.render(text
                                 if (len(text)<MAX_TEXT_LEN_DISPLAY)
                                 else (text[0:MAX_TEXT_LEN_DISPLAY]+"..."),
                                 1,
                                 TEXT_COLOR)

        size = label.get_rect()[2:4]
        self.windowSurface.blit(label, (WIDTH/2 - size[0]/2, HEIGHT/2 - size[1]/2))

        pygame.display.flip()

    def waveform_from_mic(self, key = pygame.K_SPACE) -> np.ndarray:
        logging.info("Capturing waveform from microphone")
        self.display_rec_start()

        stream = self.audio.open(format=INPUT_FORMAT,
                                 channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE,
                                 input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = []

        while True:
            pygame.event.pump() # process event queue
            pressed = pygame.key.get_pressed()
            if pressed[key]:
                data = stream.read(INPUT_CHUNK)
                frames.append(data)
            else:
                break

        stream.stop_stream()
        stream.close()

        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)

    def speech_to_text(self, waveform):
        logging.info("Converting speech to text")
        result_queue = queue.Queue()

        def transcribe_speech():
            try:
                logging.info("Starting transcription")
                transcript = self.model.transcribe(waveform,
                                                language=self.config.whisperRecognition.lang,
                                                fp16=torch.cuda.is_available())
                logging.info("Transcription completed")
                text = transcript["text"]
                print('\nMe:\n', text.strip())
                result_queue.put(text)
            except Exception as e:
                logging.error(f"An error occurred during transcription: {str(e)}")
                result_queue.put("")

        transcription_thread = threading.Thread(target=transcribe_speech)
        transcription_thread.start()
        transcription_thread.join()

        return result_queue.get()

    def handle_application_intent(self, text):
        """
        Handle opening applications
        """
        app_mapping = {
            'google': 'google-chrome',
            'chrome': 'google-chrome',
            'firefox': 'firefox',
            'terminal': 'gnome-terminal',
            'files': 'nautilus',
            'settings': 'gnome-control-center',
            'calendar': 'gnome-calendar'
        }

        for app_name, app_command in app_mapping.items():
            if app_name in text.lower():
                try:
                    subprocess.Popen([app_command])
                    return f"Opening {app_name}"
                except Exception as e:
                    return f"Could not open {app_name}: {str(e)}"
        
        return "Sorry, I couldn't identify the application to open."

    def handle_file_intent(self, text):
        """
        Handle file-related commands
        """
        # Extract path from the text
        match = re.search(r'(in|at)\s+([\'"]?.*[\'"]?)', text)
        if match:
            path = match.group(2).strip("'\"")
            path = os.path.expanduser(path) if path.startswith('~') else path
        else:
            path = os.getcwd()  # Default to current working directory

        try:
            if 'list files' in text.lower() or 'show files' in text.lower():
                try:
                    files = os.listdir(path)
                    return f"Files in {path}: {', '.join(files)}"
                except Exception as e:
                    return f"Error listing files: {str(e)}"
            
            # Check for opening a specific file
            elif 'open file' in text.lower():
                match = re.search(r'open file\s+([\'"]?.*[\'"]?)', text)
                if match:
                    filepath = match.group(1).strip("'\"")
                    filepath = os.path.expanduser(filepath) if filepath.startswith('~') else filepath
                    
                    if os.path.exists(filepath):
                        subprocess.Popen(['xdg-open', filepath])
                        return f"Opening file: {filepath}"
                    else:
                        return f"File not found: {filepath}"
        except Exception as e:
            return f"Error processing file command: {str(e)}"
        
        return "Sorry, I couldn't understand the file-related command."

    def handle_search_intent(self, text):
        """
        Handle web search commands
        """
        # Extract search query
        match = re.search(r'(search|find|look up)\s+(.*)', text, re.IGNORECASE)
        if match:
            query = match.group(2).strip()
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            try:
                subprocess.Popen(['xdg-open', search_url])
                return f"Searching Google for: {query}"
            except Exception as e:
                return f"Could not perform search: {str(e)}"
        
        return "Sorry, I couldn't understand the search command."

    def ask_ollama(self, prompt, responseCallback):
        """
        Enhanced method to handle different intents
        """
        logging.info(f"Processing prompt: {prompt}")

        # Classify intent
        intent = self.intent_classifier.classify_intent(prompt)
        logging.info(f"Detected Intent: {intent}")

        # Handle different intents
        if intent == 'application':
            response = self.handle_application_intent(prompt)
        elif intent == 'file':
            response = self.handle_file_intent(prompt)
        elif intent == 'search':
            response = self.handle_search_intent(prompt)
        else:
            # Default to regular OLLaMa query for general queries
            try:
                response = self._get_ollama_response(prompt)
            except Exception as e:
                response = f"An error occurred: {str(e)}"
        
        # Callback with the response
        responseCallback(response)

    def _get_ollama_response(self, prompt):
        """
        Get response from OLLaMa for general queries
        """
        full_prompt = prompt if hasattr(self, "contextSent") else (prompt)
        self.contextSent = True
        jsonParam = {
            "model": self.config.ollama.model,
            "stream": True,
            "context": self.context,
            "prompt": full_prompt
        }
        
        response = requests.post(
            self.config.ollama.url,
            json=jsonParam,
            headers=OLLAMA_REST_HEADERS,
            stream=True,
            timeout=30
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            body = json.loads(line)
            token = body.get('response', '')
            full_response += token

            if body.get('done', False) and 'context' in body:
                self.context = body['context']
                break

        return full_response.strip()

    def text_to_speech(self, text):
        logging.info(f"Converting text to speech: {text}")
        print('\nAI:\n', text.strip())

        def play_speech():
            try:
                logging.info("Initializing TTS engine")
                engine = pyttsx3.init()
                
                # Adjust the speech rate (optional)
                rate = engine.getProperty('rate')
                engine.setProperty('rate', rate - 50)  # Decrease the rate by 50 units
                
                # Add a short delay before converting text to speech
                time.sleep(0.5)  # Adjust the delay as needed
                
                logging.info("Converting text to speech")
                engine.say(text)
                engine.runAndWait()
                logging.info("Speech playback completed")
            except Exception as e:
                logging.error(f"An error occurred during speech playback: {str(e)}")

        speech_thread = threading.Thread(target=play_speech)
        speech_thread.start()

def main():
    logging.info("Starting Assistant")
    
    ass = Assistant()

    push_to_talk_key = pygame.K_SPACE
    quit_key = pygame.K_ESCAPE

    while True:
        ass.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == push_to_talk_key:
                    logging.info("Push-to-talk key pressed")
                    speech = ass.waveform_from_mic(push_to_talk_key)

                    transcription = ass.speech_to_text(waveform=speech)

                    ass.ask_ollama(transcription, ass.text_to_speech)

                    time.sleep(1)
                    ass.display_message(ass.config.messages.pressSpace)

                elif event.key == quit_key:
                    logging.info("Quit key pressed")
                    ass.shutdown()

if __name__ == "__main__":
    main()





'''
import sys
import json
import wave
import time
import pyttsx3
import torch
import requests
import soundfile
import yaml
import pygame
import pygame.locals
import numpy as np
import pyaudio
import whisper
import logging
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

BACK_COLOR = (0,0,0)
REC_COLOR = (255,0,0)
TEXT_COLOR = (255,255,255)
REC_SIZE = 80
FONT_SIZE = 24
WIDTH = 320
HEIGHT = 240
KWIDTH = 20
KHEIGHT = 6
MAX_TEXT_LEN_DISPLAY = 32

INPUT_DEFAULT_DURATION_SECONDS = 5
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024
OLLAMA_REST_HEADERS = {'Content-Type': 'application/json'}
INPUT_CONFIG_PATH ="assistant.yaml"

class Assistant:
    def __init__(self):
        logging.info("Initializing Assistant")
        self.config = self.init_config()

        programIcon = pygame.image.load('assistant.png')

        self.clock = pygame.time.Clock()
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("Assistant")
        self.windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.audio = pyaudio.PyAudio()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 600)  
        self.tts.setProperty('volume', 0.9) 

        try:
            self.audio.open(format=INPUT_FORMAT,
                            channels=INPUT_CHANNELS,
                            rate=INPUT_RATE,
                            input=True,
                            frames_per_buffer=INPUT_CHUNK).close()
        except Exception as e:
            logging.error(f"Error opening audio stream: {str(e)}")
            self.wait_exit()

        self.display_message(self.config.messages.loadingModel)
        self.model = whisper.load_model("base.en")
        self.context = []

        self.text_to_speech(self.config.conversation.greeting)
        time.sleep(0.5)
        self.display_message(self.config.messages.pressSpace)

    def wait_exit(self):
        while True:
            self.display_message(self.config.messages.noAudioInput)
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    self.shutdown()

    def shutdown(self):
        logging.info("Shutting down Assistant")
        self.audio.terminate()
        pygame.quit()
        sys.exit()

    def init_config(self):
        logging.info("Initializing configuration")
        class Inst:
            pass

        with open('assistant.yaml', encoding='utf-8') as data:
            configYaml = yaml.safe_load(data)

        config = Inst()
        config.messages = Inst()
        config.messages.loadingModel = configYaml["messages"]["loadingModel"]
        config.messages.pressSpace = configYaml["messages"]["pressSpace"]
        config.messages.noAudioInput = configYaml["messages"]["noAudioInput"]

        config.conversation = Inst()
        config.conversation.greeting = configYaml["conversation"]["greeting"]

        config.ollama = Inst()
        config.ollama.url = configYaml["ollama"]["url"]
        config.ollama.model = configYaml["ollama"]["model"]

        config.whisperRecognition = Inst()
        config.whisperRecognition.modelPath = configYaml["whisperRecognition"]["modelPath"]
        config.whisperRecognition.lang = configYaml["whisperRecognition"]["lang"]

        return config

    def display_rec_start(self):
        logging.info("Displaying recording start")
        self.windowSurface.fill(BACK_COLOR)
        pygame.draw.circle(self.windowSurface, REC_COLOR, (WIDTH/2, HEIGHT/2), REC_SIZE)
        pygame.display.flip()

    def display_sound_energy(self, energy):
        logging.info(f"Displaying sound energy: {energy}")
        COL_COUNT = 5
        RED_CENTER = 100
        FACTOR = 10
        MAX_AMPLITUDE = 100

        self.windowSurface.fill(BACK_COLOR)
        amplitude = int(MAX_AMPLITUDE*energy)
        hspace, vspace = 2*KWIDTH, int(KHEIGHT/2)
        def rect_coords(x, y):
            return (int(x-KWIDTH/2), int(y-KHEIGHT/2),
                    KWIDTH, KHEIGHT)
        for i in range(-int(np.floor(COL_COUNT/2)), int(np.ceil(COL_COUNT/2))):
            x, y, count = WIDTH/2+(i*hspace), HEIGHT/2, amplitude-2*abs(i)

            mid = int(np.ceil(count/2))
            for i in range(0, mid):
                offset = i*(KHEIGHT+vspace)
                pygame.draw.rect(self.windowSurface, RED_CENTER,
                                rect_coords(x, y+offset))
                #mirror:
                pygame.draw.rect(self.windowSurface, RED_CENTER,
                                rect_coords(x, y-offset))
        pygame.display.flip()

    def display_message(self, text):
        logging.info(f"Displaying message: {text}")
        self.windowSurface.fill(BACK_COLOR)

        label = self.font.render(text
                                 if (len(text)<MAX_TEXT_LEN_DISPLAY)
                                 else (text[0:MAX_TEXT_LEN_DISPLAY]+"..."),
                                 1,
                                 TEXT_COLOR)

        size = label.get_rect()[2:4]
        self.windowSurface.blit(label, (WIDTH/2 - size[0]/2, HEIGHT/2 - size[1]/2))

        pygame.display.flip()

    def waveform_from_mic(self, key = pygame.K_SPACE) -> np.ndarray:
        logging.info("Capturing waveform from microphone")
        self.display_rec_start()

        stream = self.audio.open(format=INPUT_FORMAT,
                                 channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE,
                                 input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = []

        while True:
            pygame.event.pump() # process event queue
            pressed = pygame.key.get_pressed()
            if pressed[key]:
                data = stream.read(INPUT_CHUNK)
                frames.append(data)
            else:
                break

        stream.stop_stream()
        stream.close()

        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)

    def speech_to_text(self, waveform):
        logging.info("Converting speech to text")
        result_queue = queue.Queue()

        def transcribe_speech():
            try:
                logging.info("Starting transcription")
                transcript = self.model.transcribe(waveform,
                                                language=self.config.whisperRecognition.lang,
                                                fp16=torch.cuda.is_available())
                logging.info("Transcription completed")
                text = transcript["text"]
                print('\nMe:\n', text.strip())
                result_queue.put(text)
            except Exception as e:
                logging.error(f"An error occurred during transcription: {str(e)}")
                result_queue.put("")

        transcription_thread = threading.Thread(target=transcribe_speech)
        transcription_thread.start()
        transcription_thread.join()

        return result_queue.get()


    def ask_ollama(self, prompt, responseCallback):
        logging.info(f"Asking OLLaMa with prompt: {prompt}")
        full_prompt = prompt if hasattr(self, "contextSent") else (prompt)
        self.contextSent = True
        jsonParam = {
            "model": self.config.ollama.model,
            "stream": True,
            "context": self.context,
            "prompt": full_prompt
        }
        
        try:
            response = requests.post(self.config.ollama.url,
                                    json=jsonParam,
                                    headers=OLLAMA_REST_HEADERS,
                                    stream=True,
                                    timeout=30)  # Increase the timeout value
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                body = json.loads(line)
                token = body.get('response', '')
                full_response += token

                if 'error' in body:
                    logging.error(f"Error from OLLaMa: {body['error']}")
                    responseCallback("Error: " + body['error'])
                    return

                if body.get('done', False) and 'context' in body:
                    self.context = body['context']
                    break

            responseCallback(full_response.strip())

        except requests.exceptions.ReadTimeout as e:
            logging.error(f"ReadTimeout occurred while asking OLLaMa: {str(e)}")
            responseCallback("Sorry, the request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while asking OLLaMa: {str(e)}")
            responseCallback("Sorry, an error occurred. Please try again.")


    def text_to_speech(self, text):
        logging.info(f"Converting text to speech: {text}")
        print('\nAI:\n', text.strip())

        def play_speech():
            try:
                logging.info("Initializing TTS engine")
                engine = pyttsx3.init()
                
                # Adjust the speech rate (optional)
                rate = engine.getProperty('rate')
                engine.setProperty('rate', rate - 50)  # Decrease the rate by 50 units
                
                # Add a short delay before converting text to speech
                time.sleep(0.5)  # Adjust the delay as needed
                
                logging.info("Converting text to speech")
                engine.say(text)
                engine.runAndWait()
                logging.info("Speech playback completed")
            except Exception as e:
                logging.error(f"An error occurred during speech playback: {str(e)}")

        speech_thread = threading.Thread(target=play_speech)
        speech_thread.start()


def main():
    logging.info("Starting Assistant")
    pygame.init()

    ass = Assistant()

    push_to_talk_key = pygame.K_SPACE
    quit_key = pygame.K_ESCAPE

    while True:
        ass.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == push_to_talk_key:
                    logging.info("Push-to-talk key pressed")
                    speech = ass.waveform_from_mic(push_to_talk_key)

                    transcription = ass.speech_to_text(waveform=speech)

                    ass.ask_ollama(transcription, ass.text_to_speech)

                    time.sleep(1)
                    ass.display_message(ass.config.messages.pressSpace)

                elif event.key == quit_key:
                    logging.info("Quit key pressed")
                    ass.shutdown()


if __name__ == "__main__":
    main()
'''