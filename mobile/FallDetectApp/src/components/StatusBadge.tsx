import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors } from '../theme/colors';
import type { MqttConnectionStatus } from '../types';

interface StatusBadgeProps {
  status: MqttConnectionStatus;
}

const STATUS_CONFIG: Record<MqttConnectionStatus, { label: string; color: string; bg: string }> = {
  connected: { label: 'Conectado', color: Colors.success, bg: Colors.successDim },
  connecting: { label: 'Conectando...', color: Colors.warning, bg: Colors.surfaceLight },
  disconnected: { label: 'Desconectado', color: Colors.danger, bg: Colors.dangerDim },
  error: { label: 'Erro de Conexão', color: Colors.danger, bg: Colors.dangerDim },
};

export function StatusBadge({ status }: StatusBadgeProps) {
  const config = STATUS_CONFIG[status];

  return (
    <View style={[styles.container, { backgroundColor: config.bg }]}>
      <View style={[styles.dot, { backgroundColor: config.color }]} />
      <Text style={[styles.label, { color: config.color }]}>{config.label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    alignSelf: 'flex-start',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
  },
});
