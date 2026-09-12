import tkinter as tk
from tkinter import ttk
import threading
import queue
import os

import numpy as np
import sounddevice as sd
import librosa
import joblib


# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

RMS_THRESHOLD = 0.02

audio_running = False
audio_queue = queue.Queue()


# Load ML model
try:
    model_path = os.path.join(
        os.path.dirname(__file__),
        "BabyCry_model.pkl"
    )

    model = joblib.load(model_path)
    print("Model loaded successfully")

except Exception as e:
    model = None
    print("Model error:", e)


# Calculate sound level
def calculate_rms(audio):
    return np.sqrt(np.mean(audio ** 2))


# Extract 26 features for the model
def get_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    features = np.concatenate([mean, std])

    return features.reshape(1, -1)


# Predict baby's condition
def predict(audio):

    if model is None:
        return "Model not loaded", 0

    try:
        features = get_features(audio)

        result = model.predict(features)[0]

        confidence = 0

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)
            confidence = np.max(probability[0]) * 100

        return result, confidence

    except Exception as e:
        print("Prediction error:", e)
        return "Prediction error", 0


# Record and analyze audio
def audio_monitoring():

    global audio_running

    while audio_running:

        try:
            # Record 1.5 seconds
            audio = sd.rec(
                CHUNK_SIZE,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            audio = audio.flatten()

            # Calculate RMS
            rms = calculate_rms(audio)

            audio_queue.put(("rms", rms))

            # Check if there is sound
            if rms > RMS_THRESHOLD:

                audio_queue.put(
                    ("status", "Voice / Cry detected")
                )

                # ML prediction
                condition, confidence = predict(audio)

                audio_queue.put(
                    ("prediction", condition, confidence)
                )

            else:

                audio_queue.put(
                    ("status", "No voice detected")
                )

                audio_queue.put(
                    ("prediction", "No cry", 0)
                )

        except Exception as e:

            audio_queue.put(
                ("error", str(e))
            )

            break


# Start button
def start_monitoring():

    global audio_running

    if audio_running:
        return

    audio_running = True

    status_label.config(
        text="Audio Monitoring: ON"
    )

    audio_status_label.config(
        text="Listening..."
    )

    condition_label.config(
        text="Waiting..."
    )

    confidence_label.config(
        text="Confidence: --"
    )

    start_button.config(
        state="disabled"
    )

    stop_button.config(
        state="normal"
    )

    threading.Thread(
        target=audio_monitoring,
        daemon=True
    ).start()


# Stop button
def stop_monitoring():

    global audio_running

    audio_running = False

    status_label.config(
        text="Audio Monitoring: OFF"
    )

    audio_status_label.config(
        text="Monitoring stopped"
    )

    rms_label.config(
        text="RMS: --"
    )

    condition_label.config(
        text="--"
    )

    confidence_label.config(
        text="Confidence: --"
    )

    start_button.config(
        state="normal"
    )

    stop_button.config(
        state="disabled"
    )


# Update GUI
def update_gui():

    try:

        while True:

            message = audio_queue.get_nowait()

            if message[0] == "rms":
                rms_label.config(
                    text=f"RMS: {message[1]:.4f}"
                )

            elif message[0] == "status":

                audio_status_label.config(
                    text=message[1]
                )

            elif message[0] == "prediction":

                condition_label.config(
                    text=message[1]
                )

                if message[2] > 0:

                    confidence_label.config(
                        text=f"Confidence: {message[2]:.1f}%"
                    )

            elif message[0] == "error":

                audio_status_label.config(
                    text="Audio Error"
                )

                print("Audio Error:", message[1])

    except queue.Empty:
        pass

    root.after(100, update_gui)


# =========================
# GUI
# =========================

root = tk.Tk()

root.title("Smart Nursery Guardian")

root.geometry("550x550")

root.resizable(False, False)


# Title
title_label = ttk.Label(
    root,
    text="Smart Nursery Guardian",
    font=("Arial", 20, "bold")
)

title_label.pack(pady=20)


subtitle_label = ttk.Label(
    root,
    text="Audio Monitoring System",
    font=("Arial", 13)
)

subtitle_label.pack()


# Audio section
audio_frame = ttk.LabelFrame(
    root,
    text="Audio Status",
    padding=20
)

audio_frame.pack(
    padx=30,
    pady=20,
    fill="x"
)


audio_status_label = ttk.Label(
    audio_frame,
    text="Monitoring stopped",
    font=("Arial", 15)
)

audio_status_label.pack(pady=8)


rms_label = ttk.Label(
    audio_frame,
    text="RMS: --",
    font=("Arial", 12)
)

rms_label.pack(pady=5)


# Baby condition section
condition_frame = ttk.LabelFrame(
    root,
    text="Baby Condition",
    padding=20
)

condition_frame.pack(
    padx=30,
    pady=10,
    fill="x"
)


condition_label = ttk.Label(
    condition_frame,
    text="--",
    font=("Arial", 16, "bold")
)

condition_label.pack(pady=5)


confidence_label = ttk.Label(
    condition_frame,
    text="Confidence: --",
    font=("Arial", 11)
)

confidence_label.pack(pady=5)


# Monitoring status
status_label = ttk.Label(
    root,
    text="Audio Monitoring: OFF",
    font=("Arial", 12)
)

status_label.pack(pady=10)


# Buttons
button_frame = ttk.Frame(root)

button_frame.pack(pady=20)


start_button = ttk.Button(
    button_frame,
    text="START",
    command=start_monitoring
)

start_button.grid(
    row=0,
    column=0,
    padx=15
)


stop_button = ttk.Button(
    button_frame,
    text="STOP",
    command=stop_monitoring,
    state="disabled"
)

stop_button.grid(
    row=0,
    column=1,
    padx=15
)


# Start updating GUI
root.after(100, update_gui)

root.mainloop()