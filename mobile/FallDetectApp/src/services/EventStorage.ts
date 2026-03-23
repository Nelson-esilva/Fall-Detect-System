import AsyncStorage from '@react-native-async-storage/async-storage';
import type { FallEvent, AppSettings } from '../types';
import { DEFAULT_SETTINGS } from '../types';

const EVENTS_KEY = '@fall_detect/events';
const SETTINGS_KEY = '@fall_detect/settings';

export const EventStorage = {
  async loadEvents(): Promise<FallEvent[]> {
    try {
      const raw = await AsyncStorage.getItem(EVENTS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  },

  async saveEvents(events: FallEvent[]): Promise<void> {
    try {
      await AsyncStorage.setItem(EVENTS_KEY, JSON.stringify(events.slice(0, 200)));
    } catch {
      console.warn('[Storage] Erro ao salvar eventos');
    }
  },

  async clearEvents(): Promise<void> {
    try {
      await AsyncStorage.removeItem(EVENTS_KEY);
    } catch {
      console.warn('[Storage] Erro ao limpar eventos');
    }
  },

  async loadSettings(): Promise<AppSettings> {
    try {
      const raw = await AsyncStorage.getItem(SETTINGS_KEY);
      return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  },

  async saveSettings(settings: AppSettings): Promise<void> {
    try {
      await AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      console.warn('[Storage] Erro ao salvar configurações');
    }
  },
};
