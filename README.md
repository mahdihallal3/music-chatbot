***Bobbybot - Conversational Agent

*Project Overview:

Bobbybot is a conversational agent (chatbot) developed using Python and NLP methodologies. The project involves several key NLP tasks, such as text preprocessing, vectorization, speech recognition, and text-to-speech capabilities. The goal of Bobbybot is to interact with users by responding to greetings, answering questions from a provided Wikipedia text, and offering both text-based and voice-based interaction. Bobby is tailored to answer music-related questions.


*Features:

Text Preprocessing: Load and preprocess text from a Wikipedia page (Coronavirus, Tennis, or any chosen topic).

Greeting Interaction: The bot responds to user greetings by matching inputs with a predefined list and providing random replies.

Question Answering: Users can ask questions, and the bot returns relevant text using TF-IDF vectorization and Cosine similarity.

Speech Recognition: Bobbybot can process spoken input using the SpeechRecognition package and reply based on the transcribed text.

Text-to-Speech: The bot's replies can be played back via audio using the gTTS (Google Text-to-Speech) package.

Session Termination: The bot will close the session if the user says "Bye" or "Bye Bobby" with an appropriate farewell response.


*Project Structure:

Module 1: Text Preprocessing. Preprocess and normalize text extracted from Wikipedia.

Module 2: Greeting Bot. Responds to greetings based on predefined lists of user inputs and bot responses.

Module 3: Question Answering. Uses TF-IDF vectorization and Cosine similarity to answer user questions based on the preprocessed text.

Module 4: Speech Recognition. Processes user input from both voice recordings and live microphone input, using the SpeechRecognition library.

Module 5: Text-to-Speech. Converts the bot's replies to audio using the gTTS package.

Module 6: Session Termination. Handles user input to end the conversation with an appropriate goodbye response.


*Technologies Used:

Python: Programming language used for all modules.

TF-IDF (Term Frequency-Inverse Document Frequency): For vectorizing the input text and questions.

Cosine Similarity: To find the most relevant text in response to user questions.

SpeechRecognition: Python library for speech-to-text functionality.

gTTS (Google Text-to-Speech): For converting text-to-speech output.

PortAudio & PyAudio: For handling microphone input in speech recognition.
