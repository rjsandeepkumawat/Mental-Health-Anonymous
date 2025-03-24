from transformers import pipeline
import numpy as np

class EmotionClassifier:
    def __init__(self):
        # Initialize the emotion classifier using a pre-trained model
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        
        # Define the emotions we want to detect
        self.emotions = [
            "joy", "sadness", "anger", "fear", 
            "surprise", "disgust", "neutral"
        ]

    def classify(self, text):
        try:
            # Get predictions from the model
            predictions = self.classifier(text)[0]
            
            # Find the emotion with highest confidence
            primary_emotion = max(predictions, key=lambda x: x['score'])
            
            return {
                "primary_emotion": primary_emotion['label'],
                "confidence": primary_emotion['score'],
                "all_emotions": predictions
            }
        except Exception as e:
            return {
                "primary_emotion": "neutral",
                "confidence": 1.0,
                "error": str(e)
            } 