import os
import json

# Set the directory paths for both datasets
base_dir = "/home/s6kogase/seminar/data"
ljspeech_dir = os.path.join(base_dir, "A_ljspeech")
bigvgan_dir = os.path.join(base_dir, "J_bigvgan")

# Create a list to hold the JSON objects
data = []

# Function to generate the JSON object
def generate_json_obj(audio_id, dataset, output):
    return {
        "instruction": "Closed-ended question: Analyze the audio and determine whether it is a real recording or an audio deepfake.",
        "input": "",
        "audio_id": audio_id,
        "dataset": dataset,
        "task": "cla_label_des",
        "output": output
    }

# Process files from the A_ljspeech directory (real recordings)
for root, dirs, files in os.walk(ljspeech_dir):
    for file in files:
        if file.endswith(".wav"):  # Make sure it's an audio file
            audio_path = os.path.join(root, file)
            output_text = "It is a real recording, because it is from the LJSpeech dataset."
            json_obj = generate_json_obj(audio_path, "ljspeech", output_text)
            data.append(json_obj)

# Process files from the J_bigvgan directory (audio deepfakes)
for root, dirs, files in os.walk(bigvgan_dir):
    for file in files:
        if file.endswith(".wav") or file.endswith(".flac"):  # Handle both wav and flac files
            audio_path = os.path.join(root, file)
            output_text = "It is an audio deepfake, because it shows characteristics of the BigVGAN vocoder."
            json_obj = generate_json_obj(audio_path, "ljspeech", output_text)
            data.append(json_obj)

# Write the list of JSON objects to a file
output_file = "audio_classification_data.json"
import random
random.shuffle(data)
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"JSON file generated at: {output_file}")
