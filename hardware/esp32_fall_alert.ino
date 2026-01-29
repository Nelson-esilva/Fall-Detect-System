/*
 * ESP32 - Sistema de Alerta de Quedas
 * Recebe alertas do sistema de detecção de quedas e dispara alarmes locais
 * 
 * Hardware necessário:
 * - ESP32 (qualquer variante)
 * - Buzzer passivo conectado ao GPIO 18
 * - LED vermelho conectado ao GPIO 19
 * - LED verde conectado ao GPIO 21 (status)
 * - Botão de teste conectado ao GPIO 0 (opcional)
 * 
 * Comunicação:
 * - Serial (USB): Recebe JSON via Serial
 * - WiFi/MQTT: Recebe mensagens MQTT (opcional, descomente se usar)
 * 
 * Autor: Sistema de Detecção de Quedas
 * Data: 2025
 */

// ==================== CONFIGURAÇÕES ====================
#define BUZZER_PIN 18
#define LED_RED_PIN 19
#define LED_GREEN_PIN 21
#define BUTTON_TEST_PIN 0

// Configurações MQTT (descomente se usar WiFi)
/*
#define WIFI_SSID "SEU_WIFI"
#define WIFI_PASSWORD "SUA_SENHA"
#define MQTT_BROKER "192.168.1.100"
#define MQTT_PORT 1883
#define MQTT_TOPIC "fall_detection/alerts"
*/

// Configurações do alarme
#define ALERT_DURATION_MS 10000  // Duração do alerta (10 segundos)
#define BUZZER_FREQ_ALERT 2000   // Frequência do buzzer (Hz)
#define BLINK_INTERVAL_MS 500    // Intervalo de piscada do LED (ms)

// ==================== BIBLIOTECAS ====================
#include <ArduinoJson.h>
// Para MQTT (descomente se usar):
// #include <WiFi.h>
// #include <PubSubClient.h>

// ==================== VARIÁVEIS GLOBAIS ====================
bool alert_active = false;
unsigned long alert_start_time = 0;
unsigned long last_blink = 0;
bool led_state = false;
float last_confidence = 0.0;

// Para MQTT (descomente se usar):
// WiFiClient espClient;
// PubSubClient mqtt_client(espClient);

// ==================== SETUP ====================
void setup() {
  // Inicializar Serial
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n========================================");
  Serial.println("ESP32 - Sistema de Alerta de Quedas");
  Serial.println("========================================\n");
  
  // Configurar pinos
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(BUTTON_TEST_PIN, INPUT_PULLUP);
  
  // Estado inicial
  digitalWrite(LED_GREEN_PIN, HIGH);  // LED verde ligado = sistema OK
  digitalWrite(LED_RED_PIN, LOW);
  noTone(BUZZER_PIN);
  
  Serial.println("[OK] Sistema inicializado");
  Serial.println("[INFO] Aguardando alertas...\n");
  
  // Para MQTT (descomente se usar):
  /*
  setup_wifi();
  mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt_client.setCallback(mqtt_callback);
  */
}

// ==================== LOOP PRINCIPAL ====================
void loop() {
  // Verificar Serial (USB)
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil('\n');
    message.trim();
    if (message.length() > 0) {
      process_alert(message);
    }
  }
  
  // Para MQTT (descomente se usar):
  /*
  if (!mqtt_client.connected()) {
    reconnect_mqtt();
  }
  mqtt_client.loop();
  */
  
  // Gerenciar alerta ativo
  if (alert_active) {
    handle_active_alert();
  }
  
  // Verificar botão de teste
  if (digitalRead(BUTTON_TEST_PIN) == LOW) {
    delay(50);  // Debounce
    if (digitalRead(BUTTON_TEST_PIN) == LOW) {
      trigger_test_alert();
      delay(500);
    }
  }
  
  delay(10);
}

// ==================== PROCESSAMENTO DE ALERTAS ====================
void process_alert(String json_message) {
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, json_message);
  
  if (error) {
    Serial.print("[ERRO] Falha ao parsear JSON: ");
    Serial.println(error.c_str());
    return;
  }
  
  String alert_type = doc["alert"] | "";
  float confidence = doc["confidence"] | 0.0;
  String timestamp = doc["timestamp"] | "";
  
  if (alert_type == "FALL_DETECTED") {
    Serial.println("========================================");
    Serial.println("ALERTA DE QUEDA DETECTADA!");
    Serial.println("========================================");
    Serial.print("Confianca: ");
    Serial.print(confidence * 100);
    Serial.println("%");
    Serial.print("Timestamp: ");
    Serial.println(timestamp);
    Serial.println("========================================\n");
    
    trigger_alert(confidence);
  } else if (alert_type == "TEST") {
    Serial.println("[TESTE] Alerta de teste recebido");
    trigger_test_alert();
  } else {
    Serial.print("[AVISO] Tipo de alerta desconhecido: ");
    Serial.println(alert_type);
  }
}

// ==================== TRIGGER DE ALERTAS ====================
void trigger_alert(float confidence) {
  alert_active = true;
  alert_start_time = millis();
  last_confidence = confidence;
  
  // LED verde desliga (sistema em alerta)
  digitalWrite(LED_GREEN_PIN, LOW);
  
  Serial.println("[ALERTA] Sistema de alerta ativado!");
}

void trigger_test_alert() {
  Serial.println("\n[TESTE] Disparando alerta de teste...");
  trigger_alert(0.95);
}

// ==================== GERENCIAMENTO DE ALERTA ATIVO ====================
void handle_active_alert() {
  unsigned long elapsed = millis() - alert_start_time;
  
  // Verificar se alerta expirou
  if (elapsed >= ALERT_DURATION_MS) {
    stop_alert();
    return;
  }
  
  // Buzzer (tom contínuo)
  tone(BUZZER_PIN, BUZZER_FREQ_ALERT);
  
  // LED vermelho piscando
  if (millis() - last_blink >= BLINK_INTERVAL_MS) {
    led_state = !led_state;
    digitalWrite(LED_RED_PIN, led_state ? HIGH : LOW);
    last_blink = millis();
  }
}

void stop_alert() {
  alert_active = false;
  noTone(BUZZER_PIN);
  digitalWrite(LED_RED_PIN, LOW);
  digitalWrite(LED_GREEN_PIN, HIGH);  // LED verde liga = sistema OK novamente
  
  Serial.println("[ALERTA] Sistema de alerta desativado");
  Serial.println("[INFO] Aguardando proximos alertas...\n");
}

// ==================== MQTT (OPCIONAL) ====================
/*
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Conectando ao WiFi: ");
  Serial.println(WIFI_SSID);
  
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("[OK] WiFi conectado!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

void reconnect_mqtt() {
  while (!mqtt_client.connected()) {
    Serial.print("Conectando ao MQTT broker...");
    if (mqtt_client.connect("ESP32_FallAlert")) {
      Serial.println("[OK] Conectado!");
      mqtt_client.subscribe(MQTT_TOPIC);
    } else {
      Serial.print("[ERRO] Falhou. Codigo: ");
      Serial.print(mqtt_client.state());
      Serial.println(" Tentando novamente em 5 segundos...");
      delay(5000);
    }
  }
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  process_alert(message);
}
*/
