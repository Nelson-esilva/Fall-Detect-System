"""
Script de teste para ESP32
Testa a conexão e envio de alertas sem precisar do sistema de detecção completo
"""
import sys
import os
import time
from pathlib import Path

# Adicionar diretórios ao path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from esp32_interface import create_esp32_interface

def test_esp32_connection():
    """Testa conexão e envio de alertas ao ESP32"""
    
    print("="*60)
    print("TESTE DE CONEXAO ESP32")
    print("="*60)
    
    # Configurações
    connection_type = input("\nTipo de conexao (serial/mqtt) [serial]: ").strip() or "serial"
    
    if connection_type == "serial":
        port = input("Porta Serial (ex: COM3, /dev/ttyUSB0) [COM3]: ").strip() or "COM3"
        baudrate = input("Baudrate [115200]: ").strip() or "115200"
        
        try:
            esp32 = create_esp32_interface("serial", port=port, baudrate=int(baudrate))
        except Exception as e:
            print(f"ERRO: {e}")
            return
    else:  # MQTT
        broker = input("Broker MQTT [localhost]: ").strip() or "localhost"
        port = input("Porta MQTT [1883]: ").strip() or "1883"
        topic = input("Topico [fall_detection/alerts]: ").strip() or "fall_detection/alerts"
        
        try:
            esp32 = create_esp32_interface("mqtt", broker=broker, port=int(port), topic=topic)
        except Exception as e:
            print(f"ERRO: {e}")
            return
    
    if not esp32.connected:
        print("\nERRO: Nao foi possivel conectar ao ESP32.")
        print("Verifique:")
        print("  1. ESP32 esta conectado e programado?")
        print("  2. Porta esta correta?")
        print("  3. Nenhum outro programa esta usando a porta?")
        return
    
    print("\n[OK] ESP32 conectado!")
    print("\nComandos disponiveis:")
    print("  't' - Enviar alerta de teste")
    print("  'a' - Enviar alerta de queda (confianca 0.95)")
    print("  'q' - Sair")
    
    try:
        while True:
            cmd = input("\nComando: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 't':
                print("Enviando alerta de teste...")
                success = esp32.send_test_alert()
                if success:
                    print("[OK] Alerta de teste enviado!")
                else:
                    print("[ERRO] Falha ao enviar alerta")
            elif cmd == 'a':
                print("Enviando alerta de queda...")
                success = esp32.send_alert(confidence=0.95, metadata={"test": True})
                if success:
                    print("[OK] Alerta de queda enviado!")
                else:
                    print("[ERRO] Falha ao enviar alerta")
            else:
                print("Comando invalido. Use 't', 'a' ou 'q'")
    
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuario.")
    finally:
        esp32.disconnect()
        print("\nDesconectado. Teste finalizado.")

if __name__ == "__main__":
    test_esp32_connection()
