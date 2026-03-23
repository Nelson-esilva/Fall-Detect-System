import mqtt, { MqttClient, IClientOptions } from 'mqtt';
import type { FallAlertPayload, MqttConnectionStatus } from '../types';

export type StatusCallback = (status: MqttConnectionStatus) => void;
export type MessageCallback = (payload: FallAlertPayload) => void;

class MqttService {
  private client: MqttClient | null = null;
  private statusCb: StatusCallback | null = null;
  private messageCb: MessageCallback | null = null;
  private currentTopic: string = '';

  onStatusChange(cb: StatusCallback) {
    this.statusCb = cb;
  }

  onMessage(cb: MessageCallback) {
    this.messageCb = cb;
  }

  connect(host: string, port: number, topic: string) {
    this.disconnect();
    this.currentTopic = topic;
    this.emitStatus('connecting');

    try {
      const opts: IClientOptions = {
        protocol: 'ws',
        hostname: host,
        port,
        path: '/mqtt',
        reconnectPeriod: 5000,
        connectTimeout: 10000,
        clean: true,
      };

      this.client = mqtt.connect(opts);

      this.client.on('connect', () => {
        this.emitStatus('connected');
        this.client?.subscribe(topic, { qos: 1 });
      });

      this.client.on('message', (_topic: string, message: Buffer) => {
        try {
          const payload: FallAlertPayload = JSON.parse(message.toString());
          if (payload.alert === 'FALL_DETECTED' || payload.alert === 'TEST') {
            this.messageCb?.(payload);
          }
        } catch {
          console.warn('[MQTT] Mensagem inválida recebida');
        }
      });

      this.client.on('close', () => this.emitStatus('disconnected'));
      this.client.on('error', () => this.emitStatus('error'));
      this.client.on('reconnect', () => this.emitStatus('connecting'));
    } catch (e) {
      console.warn('[MQTT] Erro ao conectar:', e);
      this.emitStatus('error');
    }
  }

  disconnect() {
    if (this.client) {
      this.client.end(true);
      this.client = null;
    }
    this.emitStatus('disconnected');
  }

  get isConnected(): boolean {
    return this.client?.connected ?? false;
  }

  private emitStatus(status: MqttConnectionStatus) {
    this.statusCb?.(status);
  }
}

export const mqttService = new MqttService();
