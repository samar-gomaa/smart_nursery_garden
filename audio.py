print("Audio file started")
import numpy as np
import sounddevice as sd
import noisereduce as nr
import librosa


# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5

RMS_THRESHOLD = 0.02
ZCR_THRESHOLD = 0.03


# Record background noise
def record_noise(duration=2):

    print("Recording background noise...")

    noise = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("Noise recording finished.")

    return noise.flatten()


# Remove background noise
def reduce_noise(audio, noise_profile):

    cleaned_audio = nr.reduce_noise(
        y=audio,
        sr=SAMPLE_RATE,
        y_noise=noise_profile,
        stationary=True
    )

    return cleaned_audio


# Calculate RMS
def get_rms(audio):

    rms = np.sqrt(np.mean(audio ** 2))

    return rms


# Calculate ZCR
def get_zcr(audio):

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(
            y=audio
        )
    )

    return zcr


# Detect voice using RMS and ZCR
def detect_voice(audio):

    rms = get_rms(audio)
    zcr = get_zcr(audio)

    if rms > RMS_THRESHOLD and zcr > ZCR_THRESHOLD:

        return True

    return False


# Get the voice part
def get_voice_segment(audio):

    frame_duration = 0.03

    frame_samples = int(
        SAMPLE_RATE * frame_duration
    )

    voice_frames = []

    for start in range(
        0,
        len(audio) - frame_samples + 1,
        frame_samples
    ):

        frame = audio[
            start:start + frame_samples
        ]

        rms = get_rms(frame)
        zcr = get_zcr(frame)

        if (
            rms > RMS_THRESHOLD
            and zcr > ZCR_THRESHOLD
        ):

            voice_frames.append(frame)

    if len(voice_frames) == 0:

        return None

    voice_audio = np.concatenate(
        voice_frames
    )

    return voice_audio


# Calculate pitch / F0
def get_pitch(audio):

    pitches, magnitudes = librosa.piptrack(
        y=audio,
        sr=SAMPLE_RATE,
        fmin=300,
        fmax=1000
    )

    pitch_values = []

    for t in range(pitches.shape[1]):

        index = np.argmax(
            magnitudes[:, t]
        )

        pitch = pitches[index, t]

        if pitch > 0:

            pitch_values.append(pitch)

    if len(pitch_values) == 0:

        return 0

    return np.mean(pitch_values)


# Calculate MFCC
def get_mfcc(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    return mfcc_mean, mfcc_std


# Extract all audio features
def extract_features(audio):

    # MFCC
    mfcc_mean, mfcc_std = get_mfcc(audio)

    # RMS
    rms = get_rms(audio)

    # ZCR
    zcr = get_zcr(audio)

    # Duration
    duration = len(audio) / SAMPLE_RATE

    # Pitch
    pitch = get_pitch(audio)

    # Combine all features
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [rms],
        [zcr],
        [duration],
        [pitch]
    ])

    return features


# Process one audio chunk
def process_audio(noise_profile):

    # Record 1.5 seconds
    recording = sd.rec(
        int(
            CHUNK_DURATION * SAMPLE_RATE
        ),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = recording.flatten()

    # Remove noise
    cleaned_audio = reduce_noise(
        audio,
        noise_profile
    )

    # Check voice activity
    if not detect_voice(cleaned_audio):

        return None

    # Get voice part
    voice_audio = get_voice_segment(
        cleaned_audio
    )

    if voice_audio is None:

        return None

    # Extract features
    features = extract_features(
        voice_audio
    )

    return features


# Test audio.py directly
if __name__ == "__main__":


    noise_profile = record_noise()

    print("\nAudio processing started.")
    print("Press Ctrl+C to stop.\n")

    try:

        while True:
            features = process_audio(
                noise_profile
                )

        if features is None:

            print("[NO VOICE]")

        else:

            print("[VOICE DETECTED]")

            print(
                "Features:"
                )

            print(
                np.round(
                    features,
                    2
                        )
                    )

            print(
                "Shape:",
                    features.shape
                    )

            print("-" * 40)

    except KeyboardInterrupt:

        print("\nStopped.")