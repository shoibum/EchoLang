# record_audio.py (GUI Version with Save As Dialog)
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog # Import filedialog
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os
from pathlib import Path
import queue
import threading

# --- Configuration ---
DEFAULT_SAMPLE_RATE = 16000
# RECORDINGS_DIR = "recordings" # No longer needed for default save path
# ---

# Global variables for recording state and data
is_recording = False
audio_queue = queue.Queue()
recording_thread = None
stream = None
recorded_audio_data = None # Store data globally after stopping

def start_recording(record_button, stop_button, status_label):
    global is_recording, audio_queue, recording_thread, stream, recorded_audio_data
    if is_recording:
        return

    # Reset queue and data holder
    audio_queue = queue.Queue()
    recorded_audio_data = None # Clear previous recording data

    try:
        device_info = sd.query_devices(kind='input')
        if not device_info:
             messagebox.showerror("Error", "No input audio device found.")
             return
        sample_rate = DEFAULT_SAMPLE_RATE
        channels = 1
        dtype = 'int16'

        print(f"Using default input device: {device_info['name']}")
        print(f"Sample Rate: {sample_rate}, Channels: {channels}, Dtype: {dtype}")

        def callback(indata, frames, time, status):
            if status: print(f"Stream Status: {status}", flush=True)
            audio_queue.put(indata.copy())

        stream = sd.InputStream( samplerate=sample_rate, channels=channels, dtype=dtype, callback=callback )
        stream.start()
        is_recording = True
        status_label.config(text="Status: Recording...")
        record_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        print("Recording started...")

    except Exception as e:
        is_recording = False
        if stream and not stream.closed: stream.stop(); stream.close()
        print(f"Error starting recording stream: {e}")
        messagebox.showerror("Recording Error", f"Could not start recording:\n{e}")
        status_label.config(text="Status: Error")
        record_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)

# --- MODIFIED STOP/SAVE FUNCTION ---
def stop_recording_and_save(record_button, stop_button, status_label):
    global is_recording, audio_queue, stream, recorded_audio_data
    if not is_recording:
        return

    print("Stopping stream...")
    if stream:
        try:
            if not stream.closed: stream.stop(); stream.close()
            print("Stream closed.")
        except Exception as e: print(f"Error stopping stream: {e}")
    else: print("Stream object not found.")

    is_recording = False
    status_label.config(text="Status: Processing...")
    # Re-enable record button immediately after stopping logic starts
    record_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED) # Keep stop disabled until next recording

    recorded_frames = []
    while not audio_queue.empty():
        recorded_frames.append(audio_queue.get())

    if not recorded_frames:
        print("No audio frames recorded.")
        status_label.config(text="Status: Stopped (No data)")
        messagebox.showwarning("Warning", "No audio was recorded.")
        return # Exit function

    try:
        # Store concatenated data globally first
        recorded_audio_data = np.concatenate(recorded_frames, axis=0)
        print(f"Processing complete. Total frames: {len(recorded_audio_data)}")
        status_label.config(text="Status: Stopped. Choose Save Location.")

        # --- Ask user where to save ---
        initial_dir = Path.cwd() # Start in current directory
        initial_file = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        filepath_to_save = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile=initial_file,
            title="Save Recording As",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        # --- End ask user ---

        if filepath_to_save: # Check if user selected a file (didn't cancel)
            try:
                # Convert data type just before saving
                if np.issubdtype(recorded_audio_data.dtype, np.floating):
                    save_data = (recorded_audio_data / np.max(np.abs(recorded_audio_data)) * 32767).astype(np.int16)
                else:
                    save_data = recorded_audio_data.astype(np.int16)

                sf.write(filepath_to_save, save_data, DEFAULT_SAMPLE_RATE, subtype='PCM_16')
                saved_message = f"Audio saved to:\n{filepath_to_save}"
                print(saved_message)
                status_label.config(text=f"Status: Saved {Path(filepath_to_save).name}")
                messagebox.showinfo("Saved", saved_message)
            except Exception as e:
                save_error_message = f"Error saving file: {e}"
                print(save_error_message)
                status_label.config(text="Status: Save Error!")
                messagebox.showerror("Save Error", save_error_message)
        else:
            # User cancelled the save dialog
            print("Save cancelled by user.")
            status_label.config(text="Status: Stopped (Save Cancelled)")

    except Exception as e:
        proc_error_message = f"Error processing recorded data: {e}"
        print(proc_error_message)
        status_label.config(text="Status: Processing Error!")
        messagebox.showerror("Processing Error", proc_error_message)

# --- END MODIFIED STOP/SAVE FUNCTION ---

def create_gui():
    # (Keep create_gui function exactly as in Response #61)
    root = tk.Tk()
    root.title("Simple Audio Recorder")
    root.geometry("300x150")

    status_label = tk.Label(root, text="Status: Ready", width=35)
    status_label.pack(pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    record_button = tk.Button(button_frame, text="Record", width=10,
                              command=lambda: start_recording(record_button, stop_button, status_label))
    record_button.pack(side=tk.LEFT, padx=10)

    stop_button = tk.Button(button_frame, text="Stop & Save", width=10, state=tk.DISABLED,
                             command=lambda: stop_recording_and_save(record_button, stop_button, status_label))
    stop_button.pack(side=tk.LEFT, padx=10)

    def on_closing():
        if is_recording:
            # Decide if we should auto-save on close or just discard
            print("Recording stopped due to window close (not saved).")
            if stream and not stream.closed:
                stream.stop()
                stream.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    # (Keep __main__ block exactly as in Response #61)
    print("----------------------------")
    print(" Simple Audio Recorder GUI  ")
    print("----------------------------")
    try:
        sd.check_input_settings()
        print(f"Default Input Device: {sd.query_devices(kind='input')['name']}")
        create_gui()
    except Exception as e:
        print(f"\nERROR: Could not initialize audio device. Please ensure a microphone is connected and permissions are granted.")
        print(f"Details: {e}")
        input("Press Enter to exit.")