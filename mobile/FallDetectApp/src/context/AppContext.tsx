import React, { createContext, useContext, useReducer, useRef, useEffect, useCallback, useState } from 'react';
import { mqttService } from '../services/MqttService';
import { alarmService } from '../services/AlarmService';
import { EventStorage } from '../services/EventStorage';
import type { AppSettings, FallAlertPayload, FallEvent, MqttConnectionStatus } from '../types';
import { DEFAULT_SETTINGS } from '../types';

// --- State ---

interface AppState {
  mqttStatus: MqttConnectionStatus;
  alarmActive: boolean;
  currentAlert: FallAlertPayload | null;
  events: FallEvent[];
  settings: AppSettings;
}

const initialState: AppState = {
  mqttStatus: 'disconnected',
  alarmActive: false,
  currentAlert: null,
  events: [],
  settings: DEFAULT_SETTINGS,
};

// --- Actions ---

type Action =
  | { type: 'MQTT_STATUS'; status: MqttConnectionStatus }
  | { type: 'ALERT_RECEIVED'; payload: FallAlertPayload }
  | { type: 'ALARM_CONFIRMED' }
  | { type: 'ALARM_DISMISSED' }
  | { type: 'UPDATE_SETTINGS'; patch: Partial<AppSettings> }
  | { type: 'LOAD_PERSISTED'; events: FallEvent[]; settings: AppSettings }
  | { type: 'CLEAR_EVENTS' };

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'MQTT_STATUS':
      return { ...state, mqttStatus: action.status };

    case 'ALERT_RECEIVED': {
      const event: FallEvent = {
        id: Date.now().toString(),
        payload: action.payload,
        receivedAt: new Date().toISOString(),
        status: 'pending',
      };
      return {
        ...state,
        alarmActive: true,
        currentAlert: action.payload,
        events: [event, ...state.events].slice(0, 200),
      };
    }

    case 'ALARM_CONFIRMED': {
      const events = [...state.events];
      const idx = events.findIndex((e) => e.status === 'pending');
      if (idx >= 0) events[idx] = { ...events[idx], status: 'confirmed' };
      return { ...state, alarmActive: false, currentAlert: null, events };
    }

    case 'ALARM_DISMISSED': {
      const events = [...state.events];
      const idx = events.findIndex((e) => e.status === 'pending');
      if (idx >= 0) events[idx] = { ...events[idx], status: 'false_alarm' };
      return { ...state, alarmActive: false, currentAlert: null, events };
    }

    case 'UPDATE_SETTINGS':
      return { ...state, settings: { ...state.settings, ...action.patch } };

    case 'LOAD_PERSISTED':
      return { ...state, events: action.events, settings: action.settings };

    case 'CLEAR_EVENTS':
      return { ...state, events: [] };

    default:
      return state;
  }
}

// --- Context ---

interface AppContextValue extends AppState {
  ready: boolean;
  connect: () => void;
  disconnect: () => void;
  confirmAlarm: () => void;
  dismissAlarm: () => void;
  testAlarm: () => void;
  updateSettings: (patch: Partial<AppSettings>) => void;
  clearEvents: () => void;
  navigationRef: React.RefObject<any>;
}

const AppContext = createContext<AppContextValue | null>(null);

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp deve ser usado dentro de AppProvider');
  return ctx;
}

// --- Provider ---

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [ready, setReady] = useState(false);
  const settingsRef = useRef(state.settings);
  const navigationRef = useRef<any>(null);

  settingsRef.current = state.settings;

  // Carregar dados persistidos ao iniciar
  useEffect(() => {
    (async () => {
      const [events, settings] = await Promise.all([
        EventStorage.loadEvents(),
        EventStorage.loadSettings(),
      ]);
      dispatch({ type: 'LOAD_PERSISTED', events, settings });
      setReady(true);
    })();
  }, []);

  // Persistir eventos quando mudam
  const prevEventsRef = useRef(state.events);
  useEffect(() => {
    if (!ready) return;
    if (prevEventsRef.current !== state.events) {
      prevEventsRef.current = state.events;
      EventStorage.saveEvents(state.events);
    }
  }, [state.events, ready]);

  // Persistir settings quando mudam
  const prevSettingsRef = useRef(state.settings);
  useEffect(() => {
    if (!ready) return;
    if (prevSettingsRef.current !== state.settings) {
      prevSettingsRef.current = state.settings;
      EventStorage.saveSettings(state.settings);
    }
  }, [state.settings, ready]);

  // Configurar callbacks MQTT
  useEffect(() => {
    mqttService.onStatusChange((status) => {
      dispatch({ type: 'MQTT_STATUS', status });
    });

    mqttService.onMessage((payload) => {
      const threshold = settingsRef.current.confidenceThreshold;
      if (payload.confidence >= threshold) {
        dispatch({ type: 'ALERT_RECEIVED', payload });
        alarmService.start();
        navigationRef.current?.navigate('Alarme');
      }
    });
  }, []);

  const connect = useCallback(() => {
    const { brokerHost, brokerPort, topic } = settingsRef.current;
    mqttService.connect(brokerHost, brokerPort, topic);
  }, []);

  const disconnect = useCallback(() => {
    mqttService.disconnect();
  }, []);

  const confirmAlarm = useCallback(() => {
    alarmService.stop();
    dispatch({ type: 'ALARM_CONFIRMED' });
  }, []);

  const dismissAlarm = useCallback(() => {
    alarmService.stop();
    dispatch({ type: 'ALARM_DISMISSED' });
  }, []);

  const testAlarm = useCallback(() => {
    const testPayload: FallAlertPayload = {
      alert: 'TEST',
      confidence: 0.95,
      timestamp: new Date().toISOString(),
      metadata: { type: 'test' },
    };
    dispatch({ type: 'ALERT_RECEIVED', payload: testPayload });
    alarmService.start();
    navigationRef.current?.navigate('Alarme');
  }, []);

  const updateSettings = useCallback((patch: Partial<AppSettings>) => {
    dispatch({ type: 'UPDATE_SETTINGS', patch });
  }, []);

  const clearEvents = useCallback(() => {
    dispatch({ type: 'CLEAR_EVENTS' });
    EventStorage.clearEvents();
  }, []);

  return (
    <AppContext.Provider
      value={{
        ...state,
        ready,
        connect,
        disconnect,
        confirmAlarm,
        dismissAlarm,
        testAlarm,
        updateSettings,
        clearEvents,
        navigationRef,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
