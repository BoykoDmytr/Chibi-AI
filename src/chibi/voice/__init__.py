"""Phase 4: voice - speech synthesis (TTS), speech recognition (STT), push-to-talk.

Heavy engines (Piper, faster-whisper, PortAudio) are optional extras and are
imported lazily, so the app runs without them; this package's pure-Python
parts (text normalisation, voice activity detection, the conversation state
machine) are always importable.
"""
