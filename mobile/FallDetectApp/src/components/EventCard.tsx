import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '../theme/colors';
import type { FallEvent } from '../types';

interface EventCardProps {
  event: FallEvent;
}

const STATUS_ICON: Record<string, { name: keyof typeof Ionicons.glyphMap; color: string }> = {
  confirmed: { name: 'warning', color: Colors.danger },
  false_alarm: { name: 'close-circle', color: Colors.textMuted },
  pending: { name: 'alert-circle', color: Colors.warning },
};

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  const day = date.getDate().toString().padStart(2, '0');
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  return `${day}/${month}, ${hours}:${minutes}`;
}

const STATUS_LABELS: Record<string, string> = {
  confirmed: 'Queda Confirmada',
  false_alarm: 'Falso Alarme',
  pending: 'Não Respondido',
};

export function EventCard({ event }: EventCardProps) {
  const icon = STATUS_ICON[event.status] ?? STATUS_ICON.pending;
  const confidence = (event.payload.confidence * 100).toFixed(0);
  const isTest = event.payload.metadata?.type === 'test';

  return (
    <View style={styles.container}>
      <View style={[styles.iconContainer, { backgroundColor: icon.color + '20' }]}>
        <Ionicons name={icon.name} size={22} color={icon.color} />
      </View>
      <View style={styles.content}>
        <Text style={styles.title}>
          {isTest ? 'Teste de Alarme' : STATUS_LABELS[event.status]}
        </Text>
        <Text style={styles.subtitle}>
          {formatDate(event.receivedAt)} · Confiança: {confidence}%
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  content: {
    flex: 1,
  },
  title: {
    color: Colors.textPrimary,
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 3,
  },
  subtitle: {
    color: Colors.textSecondary,
    fontSize: 12,
  },
});
