import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import wave
import contextlib
import collections
import webrtcvad
from pydub import AudioSegment
from faster_whisper import WhisperModel

# =========================================================
# CONFIG
# =========================================================
INPUT_WAV = "output/audio.wav"

BASE_OUTPUT_DIR = "output"
JP_DIR = os.path.join(BASE_OUTPUT_DIR, "japanese_dialogue")
VOCAL_EFFECT_DIR = os.path.join(BASE_OUTPUT_DIR, "character_vocal_effect")
OTHER_DIR = os.path.join(BASE_OUTPUT_DIR, "other_or_effect")
TEMP_DIR = os.path.join(BASE_OUTPUT_DIR, "temp_segments")

os.makedirs(JP_DIR, exist_ok=True)
os.makedirs(VOCAL_EFFECT_DIR, exist_ok=True)
os.makedirs(OTHER_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

SAMPLE_RATE = 16000
FRAME_MS = 30
AGGRESSIVENESS = 0
PADDING_MS = 600

TRIGGER_RATIO = 0.50
RELEASE_RATIO = 0.90

TARGET_LANGUAGE = "ja"
MIN_LANGUAGE_PROB = 0.45
WHISPER_MODEL_SIZE = "tiny"

PRINT_DEBUG = True
KEEP_TEMP = False

# =========================================================
# MODEL
# =========================================================
model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


# =========================================================
# HELPERS
# =========================================================
def clear_wavs(folder):
    for f in os.listdir(folder):
        if f.lower().endswith(".wav"):
            os.remove(os.path.join(folder, f))


def prepare_audio_to_wav_16k_mono(src_path):
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)

    if audio.dBFS != float("-inf"):
        change_in_dBFS = -20.0 - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

    audio.export(src_path, format="wav")


def read_wave(path):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()

        if num_channels != 1:
            raise ValueError(f"Expected mono audio, got {num_channels}")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM, got {sample_width}")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate}")

        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame:
    def __init__(self, bytes_data, timestamp, duration):
        self.bytes = bytes_data
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(audio_bytes, sample_rate, frame_ms):
    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (bytes_per_frame / 2) / sample_rate

    while offset + bytes_per_frame <= len(audio_bytes):
        chunk = audio_bytes[offset:offset + bytes_per_frame]
        yield Frame(chunk, timestamp, duration)
        timestamp += duration
        offset += bytes_per_frame


def is_valid_speech_frame(vad, frame, sample_rate):
    return vad.is_speech(frame.bytes, sample_rate)


def collect_raw_segments(frames, vad, sample_rate):
    num_padding_frames = max(1, PADDING_MS // FRAME_MS)
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    voiced_frames = []
    segments = []

    for frame in frames:
        speech_flag = is_valid_speech_frame(vad, frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, speech_flag))
            num_voiced = sum(1 for _, flag in ring_buffer if flag)

            if len(ring_buffer) == ring_buffer.maxlen and num_voiced >= TRIGGER_RATIO * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(f for f, _ in ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, speech_flag))
            num_unvoiced = sum(1 for _, flag in ring_buffer if not flag)

            if len(ring_buffer) == ring_buffer.maxlen and num_unvoiced >= RELEASE_RATIO * ring_buffer.maxlen:
                start_time = voiced_frames[0].timestamp
                end_time = voiced_frames[-1].timestamp + voiced_frames[-1].duration
                segments.append((start_time, end_time, voiced_frames[:]))
                triggered = False
                voiced_frames = []
                ring_buffer.clear()

    if voiced_frames:
        start_time = voiced_frames[0].timestamp
        end_time = voiced_frames[-1].timestamp + voiced_frames[-1].duration
        segments.append((start_time, end_time, voiced_frames[:]))

    return segments


def detect_language_and_text(wav_path):
    segments, info = model.transcribe(
        wav_path,
        beam_size=1,
        vad_filter=False,
        language=None,
        task="transcribe"
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return info.language, info.language_probability, text


def classify_segment_with_whisper(wav_path):
    try:
        lang, prob, text = detect_language_and_text(wav_path)
        is_jp = (lang == TARGET_LANGUAGE and prob >= MIN_LANGUAGE_PROB)

        return {
            "lang": lang,
            "prob": prob,
            "text": text,
            "is_japanese_dialogue": is_jp
        }
    except Exception as e:
        return {
            "lang": "unknown",
            "prob": 0.0,
            "text": "",
            "is_japanese_dialogue": False,
            "error": str(e)
        }


def sanitize_text_for_filename(text, max_len=30):
    if not text:
        return "no_text"
    safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in text)
    safe = "_".join(safe.split())
    return safe[:max_len] if safe else "no_text"


def is_character_vocal_effect(result):
    text = (result.get("text") or "").strip().lower()
    lang = result.get("lang", "")
    prob = result.get("prob", 0.0)

    vocal_words = {
        "ah", "oh", "eh", "ugh", "huh", "haha", "ha", "aa", "aaa",
        "mm", "mmm", "hn", "hah", "ow", "wah", "woah", "whoa",
        "ehh", "ahh", "ohh", "ughh", "yah", "yeah", "hahaa"
    }

    if not text:
        return False

    cleaned = (
        text.replace("!", "")
            .replace("?", "")
            .replace(".", "")
            .replace(",", "")
            .replace("…", "")
            .strip()
    )
    words = cleaned.split()

    if cleaned in vocal_words:
        return True

    if len(words) <= 2 and all(w in vocal_words for w in words):
        return True

    if lang != "ja" and prob < 0.70 and len(cleaned) <= 12:
        return True

    return False


def export_and_sort_segments(wav_path, raw_segments):
    audio = AudioSegment.from_wav(wav_path)

    clear_wavs(JP_DIR)
    clear_wavs(VOCAL_EFFECT_DIR)
    clear_wavs(OTHER_DIR)
    if not KEEP_TEMP:
        clear_wavs(TEMP_DIR)

    jp_count = 0
    vocal_count = 0
    other_count = 0

    for idx, (start, end, frames) in enumerate(raw_segments, start=1):
        if not frames:
            continue

        clip = audio[int(start * 1000):int(end * 1000)]
        temp_path = os.path.join(TEMP_DIR, f"segment_{idx:03d}.wav")
        clip.export(temp_path, format="wav")

        result = classify_segment_with_whisper(temp_path)

        if PRINT_DEBUG:
            duration_ms = int((end - start) * 1000)
            print(
                f"[SEGMENT {idx:03d}] "
                f"{start:.2f}s - {end:.2f}s | "
                f"dur={duration_ms}ms | "
                f"lang={result['lang']} ({result['prob']:.2f}) | "
                f"text={result['text'][:60]!r}"
            )

        text_tag = sanitize_text_for_filename(result["text"])

        if result["is_japanese_dialogue"]:
            jp_count += 1
            out_path = os.path.join(
                JP_DIR,
                f"jp_{jp_count:03d}_{text_tag}.wav"
            )
            clip.export(out_path, format="wav")
            print(f"Saved JP: {out_path}")

        elif is_character_vocal_effect(result):
            vocal_count += 1
            out_path = os.path.join(
                VOCAL_EFFECT_DIR,
                f"vocal_{vocal_count:03d}_{result['lang']}_{text_tag}.wav"
            )
            clip.export(out_path, format="wav")
            print(f"Saved VOCAL EFFECT: {out_path}")

        else:
            other_count += 1
            out_path = os.path.join(
                OTHER_DIR,
                f"other_{other_count:03d}_{result['lang']}_{text_tag}.wav"
            )
            clip.export(out_path, format="wav")
            print(f"Saved OTHER/EFFECT: {out_path}")

        if not KEEP_TEMP and os.path.exists(temp_path):
            os.remove(temp_path)

    return jp_count, vocal_count, other_count


def run_pipeline(input_wav):
    prepare_audio_to_wav_16k_mono(input_wav)

    audio_bytes, sample_rate = read_wave(input_wav)
    frames = list(frame_generator(audio_bytes, sample_rate, FRAME_MS))

    vad = webrtcvad.Vad(AGGRESSIVENESS)

    raw_segments = collect_raw_segments(frames, vad, sample_rate)

    print(f"\nRaw segments: {len(raw_segments)}\n")

    jp_count, vocal_count, other_count = export_and_sort_segments(input_wav, raw_segments)

    print(f"\nDone.")
    print(f"Japanese dialogue saved: {jp_count}")
    print(f"Character vocal effect saved: {vocal_count}")
    print(f"Other/effect saved: {other_count}")


if __name__ == "__main__":
    run_pipeline(INPUT_WAV)
