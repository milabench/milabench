"""Pinned English inputs of the TTS / voice-assistant benchmarks: the same text for every model, in a fixed
order (request i reads item i % len). Plain words and spelled-out numbers, so the round-trip word error
rate (Whisper transcript vs this text) measures the speech, not number or symbol normalisation.

SHORT       one-sentence assistant replies (~3-6 s of speech): latency, time to first audio
PARAGRAPHS  narration (~25-35 s of speech): long-form real-time factor, codec streaming
QUESTIONS   user turns of the voice-assistant loop (omni models): the model answers aloud
"""

SHORT = [
    "Sure, I have set a timer for twenty minutes.",
    "The weather today will be sunny with a light breeze from the west.",
    "Your meeting with the design team starts at three in the afternoon.",
    "I could not find that song, would you like me to play something similar?",
    "The kitchen lights are now off, and the front door is locked.",
    "It takes about forty minutes to drive to the airport right now.",
    "Here is a quick reminder to water the plants before you leave.",
    "The package was delivered this morning and left at the side entrance.",
    "Paris is the capital of France and sits on the river Seine.",
    "I added milk, eggs, and fresh bread to your shopping list.",
    "Good morning, you have two new messages and one missed call.",
    "The museum opens at ten and closes at six on weekdays.",
]

PARAGRAPHS = [
    "The old lighthouse stood at the edge of the cliff for more than a century. Every night its keeper climbed "
    "the narrow stairs, cleaned the great lens, and lit the lamp that guided fishing boats back to the harbor. "
    "When the storms came in from the north, he stayed awake until dawn, watching the beam sweep across the "
    "dark water and listening to the waves break against the rocks below.",

    "To make a simple loaf of bread, mix flour, water, salt, and a small amount of yeast in a large bowl. Knead "
    "the dough for about ten minutes until it feels smooth and elastic, then cover it and let it rise in a warm "
    "place for an hour. Shape the loaf, let it rest again, and bake it in a hot oven until the crust is golden "
    "and the bottom sounds hollow when you tap it.",

    "Bees play an important role in farming because they carry pollen from one flower to the next. Without "
    "them, many fruits and vegetables would produce much smaller harvests. A single hive can hold tens of "
    "thousands of workers, and each of them visits hundreds of flowers on a sunny day. Planting a variety of "
    "flowers in a garden is one of the easiest ways to help them thrive.",

    "The train left the station just after sunrise and followed the river through a wide green valley. "
    "Passengers watched small villages pass by the windows, each with a church tower and a few farms scattered "
    "across the hills. By noon the landscape had changed to tall pine forests and snowy peaks, and the air "
    "inside the carriage grew noticeably cooler as the track climbed toward the mountain pass.",
]

QUESTIONS = [
    "What is a good way to fall asleep faster at night?",
    "Can you explain in two sentences why the sky is blue?",
    "Give me three ideas for a quick and healthy breakfast.",
    "How do I keep basil plants alive on a windowsill?",
    "What should I pack for a weekend hiking trip in the mountains?",
    "Tell me a short fun fact about octopuses.",
    "How long should I boil an egg to get a soft yolk?",
    "What is the difference between weather and climate?",
]

# The voice-assistant system prompt and the instruction that turns a question into the spoken user turn
# (--audio-input self: the omni model reads the question aloud once, before the timed steps).
ASSISTANT_SYSTEM = "You are a friendly voice assistant. Answer in two or three short spoken sentences."
READ_ALOUD = "Read the following sentence aloud, word for word, and say nothing else: "

SETS = {"short": SHORT, "paragraph": PARAGRAPHS, "all": SHORT + PARAGRAPHS, "questions": QUESTIONS}
