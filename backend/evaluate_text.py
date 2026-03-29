import json
from transformers import pipeline

def evaluate_text_model():
    print("==================================================")
    print(" DeepTruth AI - Text Model Accuracy Evaluation")
    print("==================================================")
    print("Loading the highly accurate RoBERTa ChatGPT Detector...\n")
    
    detector = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")
    
    test_cases = [
        {"text": "Hey man, do you want to grab coffee later? I'm free around 4pm after my meeting.", "label": "Human"},
        {"text": "As an AI language model, I do not have personal feelings, but I can assist you with understanding complex topics.", "label": "ChatGPT"},
        {"text": "Look, I know neural networks are cool, but the way they randomly hallucinate facts is super annoying sometimes.", "label": "Human"},
        {"text": "Deepfake detection is the process of identifying AI-generated or manipulated media using specialized neural architectures to flag anomalies.", "label": "ChatGPT"},
        {"text": "I can't believe how fast my commute was today. The traffic on I-95 just completely disappeared.", "label": "Human"},
        {"text": "Furthermore, it is important to consider the socioeconomic factors that inevitably influence these statistical paradigms over time.", "label": "ChatGPT"}
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases):
        # Truncate to ensure no limits are hit
        analysis_text = case['text'][:2000]
        result = detector(analysis_text)[0]
        
        predicted_label = "ChatGPT" if result['label'] == 'ChatGPT' else 'Human'
        confidence = result['score'] if predicted_label == 'ChatGPT' else 1.0 - result['score']
        
        is_correct = predicted_label == case["label"]
        if is_correct: correct += 1
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        
        print(f"Sample {i+1}: {case['text'][:60]}...")
        print(f" - Actual: {case['label']} | Predicted: {predicted_label} | Score: {confidence:.2f}")
        print(f" - Status: {status}\n")

    accuracy = (correct / total) * 100
    print("==================================================")
    print(f" TEXT EVALUATION COMPLETE ")
    print(f" TOTAL ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    print("==================================================")
    print("Note on Audio Evaluation: Audio uses Wav2Vec 2.0. To benchmark its exact accuracy,")
    print("you must use a validation folder separated into Fake/Real. You can use evaluate_audio.py if created.")

if __name__ == "__main__":
    evaluate_text_model()
