import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Linking } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '../theme/colors';
import { useApp } from '../context/AppContext';

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  const hours = d.getHours().toString().padStart(2, '0');
  const minutes = d.getMinutes().toString().padStart(2, '0');
  const seconds = d.getSeconds().toString().padStart(2, '0');
  return `${hours}:${minutes}:${seconds}`;
}

export function AlarmScreen() {
  const { alarmActive, currentAlert, confirmAlarm, dismissAlarm, settings } = useApp();

  if (!alarmActive || !currentAlert) {
    return (
      <View style={styles.container}>
        <View style={styles.idleContent}>
          <View style={styles.idleIcon}>
            <Ionicons name="shield-checkmark" size={48} color={Colors.success} />
          </View>
          <Text style={styles.idleTitle}>Tudo Normal</Text>
          <Text style={styles.idleSubtitle}>
            Nenhum alerta de queda ativo.{'\n'}O sistema está monitorando.
          </Text>
        </View>
      </View>
    );
  }

  const confidence = (currentAlert.confidence * 100).toFixed(0);
  const isTest = currentAlert.metadata?.type === 'test';

  const handleConfirm = () => {
    confirmAlarm();
  };

  const handleDismiss = () => {
    dismissAlarm();
  };

  const handleEmergencyCall = () => {
    Linking.openURL(`tel:${settings.emergencyNumber}`);
  };

  return (
    <View style={[styles.container, styles.alarmBg]}>
      <View style={styles.alarmContent}>
        {/* Ícone de alarme */}
        <View style={styles.alarmIconOuter}>
          <View style={styles.alarmIconInner}>
            <Ionicons name="notifications" size={56} color={Colors.danger} />
          </View>
        </View>

        {/* Texto principal */}
        <Text style={styles.alarmTitle}>
          {isTest ? 'TESTE DE\nALARME' : 'QUEDA\nDETECTADA!'}
        </Text>
        <Text style={styles.alarmConfidence}>Confiança: {confidence}%</Text>
        <Text style={styles.alarmTime}>{formatTimestamp(currentAlert.timestamp)}</Text>

        {/* Botões de ação */}
        <View style={styles.actions}>
          <TouchableOpacity style={styles.confirmButton} onPress={handleConfirm} activeOpacity={0.8}>
            <Ionicons name="checkmark-circle" size={20} color={Colors.textPrimary} />
            <Text style={styles.confirmButtonText}>Confirmar Queda</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.dismissButton} onPress={handleDismiss} activeOpacity={0.8}>
            <Ionicons name="close-circle" size={20} color={Colors.textSecondary} />
            <Text style={styles.dismissButtonText}>Falso Alarme</Text>
          </TouchableOpacity>

          {!isTest && (
            <TouchableOpacity style={styles.emergencyButton} onPress={handleEmergencyCall} activeOpacity={0.8}>
              <Ionicons name="call" size={20} color={Colors.textPrimary} />
              <Text style={styles.emergencyButtonText}>Ligar {settings.emergencyNumber}</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  alarmBg: {
    backgroundColor: '#1A0A0A',
  },
  // Estado inativo
  idleContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  idleIcon: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: Colors.successDim,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  idleTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 8,
  },
  idleSubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
  },
  // Estado de alarme ativo
  alarmContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  alarmIconOuter: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: Colors.dangerDim,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 32,
  },
  alarmIconInner: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: Colors.danger + '30',
    alignItems: 'center',
    justifyContent: 'center',
  },
  alarmTitle: {
    fontSize: 32,
    fontWeight: '800',
    color: Colors.textPrimary,
    textAlign: 'center',
    marginBottom: 8,
    lineHeight: 40,
  },
  alarmConfidence: {
    fontSize: 16,
    color: Colors.textSecondary,
    marginBottom: 4,
  },
  alarmTime: {
    fontSize: 14,
    color: Colors.textMuted,
    marginBottom: 48,
  },
  actions: {
    width: '100%',
    gap: 12,
  },
  confirmButton: {
    backgroundColor: Colors.primary,
    borderRadius: 14,
    paddingVertical: 18,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
  },
  confirmButtonText: {
    color: Colors.textPrimary,
    fontSize: 16,
    fontWeight: '700',
  },
  dismissButton: {
    backgroundColor: Colors.surfaceLight,
    borderRadius: 14,
    paddingVertical: 18,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  dismissButtonText: {
    color: Colors.textSecondary,
    fontSize: 16,
    fontWeight: '600',
  },
  emergencyButton: {
    backgroundColor: Colors.danger,
    borderRadius: 14,
    paddingVertical: 18,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
  },
  emergencyButtonText: {
    color: Colors.textPrimary,
    fontSize: 16,
    fontWeight: '700',
  },
});
