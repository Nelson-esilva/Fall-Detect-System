import { Audio } from 'expo-av';
import * as Haptics from 'expo-haptics';

const alarmSound = require('../../assets/alarm.wav');

class AlarmService {
  private sound: Audio.Sound | null = null;
  private vibrationInterval: ReturnType<typeof setInterval> | null = null;
  private isPlaying = false;

  async start() {
    if (this.isPlaying) return;
    this.isPlaying = true;

    this.vibrationInterval = setInterval(() => {
      if (this.isPlaying) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      }
    }, 800);

    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: true,
        shouldDuckAndroid: false,
      });

      const { sound } = await Audio.Sound.createAsync(alarmSound, {
        isLooping: true,
        volume: 1.0,
      });
      this.sound = sound;
      await sound.playAsync();
    } catch (e) {
      console.warn('[Alarm] Erro ao reproduzir som:', e);
    }
  }

  async stop() {
    this.isPlaying = false;

    if (this.vibrationInterval) {
      clearInterval(this.vibrationInterval);
      this.vibrationInterval = null;
    }

    if (this.sound) {
      try {
        await this.sound.stopAsync();
        await this.sound.unloadAsync();
      } catch { /* already unloaded */ }
      this.sound = null;
    }
  }
}

export const alarmService = new AlarmService();
