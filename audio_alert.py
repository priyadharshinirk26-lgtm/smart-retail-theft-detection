import pygame
import numpy as np
import threading

# Initialize mixer once
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

class AudioAlert:
    def __init__(self):
        # This attribute must exist before play_beep is called
        self.muted = False

    def play_beep(self, mode="alert"):
        if self.muted:
            return

        def play():
            sample_rate = 22050
            duration = 0.3
            if mode == "alert":
                freq = 800
            elif mode == "warning":
                freq = 1000
            else:
                freq = 800

            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(freq * t * 2 * np.pi)
            sound = (tone * 32767 / np.max(np.abs(tone))).astype(np.int16)
            stereo = np.repeat(sound[:, np.newaxis], 2, axis=1)
            pygame.mixer.Sound(buffer=stereo.tobytes()).play()

        threading.Thread(target=play, daemon=True).start()

    def toggle_mute(self):
        self.muted = not self.muted
        print("Muted" if self.muted else "Unmuted")
