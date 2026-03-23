export interface FallAlertPayload {
  alert: 'FALL_DETECTED' | 'TEST';
  confidence: number;
  timestamp: string;
  metadata: {
    frame_id?: number;
    model?: string;
    type?: string;
  };
}

export type EventStatus = 'confirmed' | 'false_alarm' | 'pending';

export interface FallEvent {
  id: string;
  payload: FallAlertPayload;
  receivedAt: string;
  status: EventStatus;
}

export type MqttConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error';

export interface AppSettings {
  brokerHost: string;
  brokerPort: number;
  topic: string;
  emergencyNumber: string;
  confidenceThreshold: number;
  alarmVolume: number;
  notificationsEnabled: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  brokerHost: '192.168.1.100',
  brokerPort: 9001,
  topic: 'fall_detection/alerts',
  emergencyNumber: '192',
  confidenceThreshold: 0.7,
  alarmVolume: 1.0,
  notificationsEnabled: true,
};
