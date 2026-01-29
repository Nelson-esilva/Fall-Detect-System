"""
Módulo de Integração com ESP32
Comunicação para disparar alertas quando queda é detectada
Suporta comunicação via Serial (USB) e WiFi/MQTT
"""
import time
import json
from typing import Optional, Callable
from datetime import datetime

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("AVISO: pyserial nao instalado. Instale com: pip install pyserial")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("AVISO: paho-mqtt nao instalado. Instale com: pip install paho-mqtt")


class ESP32Interface:
    """
    Interface para comunicação com ESP32
    Suporta comunicação via Serial (USB) e MQTT (WiFi)
    """
    
    def __init__(self, connection_type: str = "serial", **kwargs):
        """
        Inicializa a interface com ESP32
        
        Args:
            connection_type: "serial" ou "mqtt"
            **kwargs: Parâmetros específicos da conexão
                - Para serial: port, baudrate (padrão: 'COM3', 115200)
                - Para MQTT: broker, port, topic (padrão: 'localhost', 1883, 'fall_detection/alerts')
        """
        self.connection_type = connection_type.lower()
        self.connected = False
        self.serial_conn = None
        self.mqtt_client = None
        self.last_alert_time = 0
        self.alert_cooldown = 5.0  # Segundos entre alertas (evita spam)
        self.alert_callback: Optional[Callable] = None
        
        if self.connection_type == "serial":
            self._init_serial(**kwargs)
        elif self.connection_type == "mqtt":
            self._init_mqtt(**kwargs)
        else:
            raise ValueError(f"Tipo de conexao invalido: {connection_type}. Use 'serial' ou 'mqtt'")
    
    def _init_serial(self, port: str = "COM3", baudrate: int = 115200, timeout: float = 1.0):
        """Inicializa conexão serial com ESP32"""
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial nao esta instalado. Execute: pip install pyserial")
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        try:
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout
            )
            time.sleep(2)  # Aguardar ESP32 inicializar
            self.connected = True
            print(f"[ESP32] Conectado via Serial em {port} ({baudrate} baud)")
        except serial.SerialException as e:
            print(f"[ESP32] ERRO ao conectar via Serial: {e}")
            print(f"[ESP32] Verifique se o ESP32 esta conectado em {port}")
            self.connected = False
    
    def _init_mqtt(self, broker: str = "localhost", port: int = 1883, 
                   topic: str = "fall_detection/alerts", client_id: str = "fall_detector_pc"):
        """Inicializa conexão MQTT com ESP32"""
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt nao esta instalado. Execute: pip install paho-mqtt")
        
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        
        try:
            self.mqtt_client = mqtt.Client(client_id=client_id)
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            self.mqtt_client.connect(broker, port, keepalive=60)
            self.mqtt_client.loop_start()
            
            # Aguardar conexão
            time.sleep(1)
            if self.connected:
                print(f"[ESP32] Conectado via MQTT em {broker}:{port} (topico: {topic})")
        except Exception as e:
            print(f"[ESP32] ERRO ao conectar via MQTT: {e}")
            self.connected = False
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback quando conecta ao broker MQTT"""
        if rc == 0:
            self.connected = True
            print("[ESP32] Conectado ao broker MQTT")
        else:
            print(f"[ESP32] Falha ao conectar MQTT. Codigo: {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback quando desconecta do broker MQTT"""
        self.connected = False
        print("[ESP32] Desconectado do broker MQTT")
    
    def send_alert(self, confidence: float = 1.0, metadata: Optional[dict] = None) -> bool:
        """
        Envia alerta de queda para o ESP32
        
        Args:
            confidence: Confiança da detecção (0.0 a 1.0)
            metadata: Metadados adicionais (timestamp, frame_id, etc.)
        
        Returns:
            True se alerta foi enviado com sucesso
        """
        # Verificar cooldown (evitar spam)
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        if not self.connected:
            print("[ESP32] AVISO: Nao conectado. Alerta nao enviado.")
            return False
        
        # Preparar mensagem
        alert_data = {
            "alert": "FALL_DETECTED",
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        success = False
        
        if self.connection_type == "serial":
            success = self._send_serial(alert_data)
        elif self.connection_type == "mqtt":
            success = self._send_mqtt(alert_data)
        
        if success:
            self.last_alert_time = current_time
            if self.alert_callback:
                self.alert_callback(alert_data)
        
        return success
    
    def _send_serial(self, data: dict) -> bool:
        """Envia dados via Serial"""
        try:
            message = json.dumps(data) + "\n"
            self.serial_conn.write(message.encode('utf-8'))
            self.serial_conn.flush()
            print(f"[ESP32] Alerta enviado via Serial: {data['alert']} (confianca: {data['confidence']:.2%})")
            return True
        except Exception as e:
            print(f"[ESP32] ERRO ao enviar via Serial: {e}")
            self.connected = False
            return False
    
    def _send_mqtt(self, data: dict) -> bool:
        """Envia dados via MQTT"""
        try:
            message = json.dumps(data)
            result = self.mqtt_client.publish(self.topic, message, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[ESP32] Alerta enviado via MQTT: {data['alert']} (confianca: {data['confidence']:.2%})")
                return True
            else:
                print(f"[ESP32] ERRO ao publicar MQTT. Codigo: {result.rc}")
                return False
        except Exception as e:
            print(f"[ESP32] ERRO ao enviar via MQTT: {e}")
            return False
    
    def send_test_alert(self) -> bool:
        """Envia alerta de teste para verificar conexão"""
        return self.send_alert(confidence=0.95, metadata={"type": "test"})
    
    def set_alert_callback(self, callback: Callable):
        """Define callback chamado quando alerta é enviado"""
        self.alert_callback = callback
    
    def disconnect(self):
        """Desconecta do ESP32"""
        if self.connection_type == "serial" and self.serial_conn:
            self.serial_conn.close()
            self.connected = False
            print("[ESP32] Desconectado (Serial)")
        elif self.connection_type == "mqtt" and self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.connected = False
            print("[ESP32] Desconectado (MQTT)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# Função auxiliar para criar interface facilmente
def create_esp32_interface(connection_type: str = "serial", **kwargs) -> ESP32Interface:
    """
    Cria uma interface ESP32 com configurações padrão
    
    Exemplos:
        # Serial (USB)
        esp = create_esp32_interface("serial", port="COM3")
        
        # MQTT (WiFi)
        esp = create_esp32_interface("mqtt", broker="192.168.1.100", topic="fall/alerts")
    """
    return ESP32Interface(connection_type=connection_type, **kwargs)
