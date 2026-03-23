import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '../theme/colors';
import { StatusBadge } from '../components/StatusBadge';
import { useApp } from '../context/AppContext';

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  const day = d.getDate().toString().padStart(2, '0');
  const month = (d.getMonth() + 1).toString().padStart(2, '0');
  const hours = d.getHours().toString().padStart(2, '0');
  const minutes = d.getMinutes().toString().padStart(2, '0');
  return `${day}/${month} – ${hours}:${minutes}`;
}

export function DashboardScreen() {
  const { mqttStatus, events, connect, disconnect, testAlarm } = useApp();
  const isConnected = mqttStatus === 'connected';
  const lastEvent = events[0] ?? null;

  const todayCount = events.filter((e) => {
    const d = new Date(e.receivedAt);
    const now = new Date();
    return d.toDateString() === now.toDateString();
  }).length;

  const weekCount = events.filter((e) => {
    const d = new Date(e.receivedAt);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    return diff < 7 * 24 * 60 * 60 * 1000;
  }).length;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.header}>Dashboard</Text>

      {/* Status de Conexão */}
      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Status de Conexão</Text>
        <View style={styles.statusRow}>
          <StatusBadge status={mqttStatus} />
          <TouchableOpacity
            style={[styles.connectBtn, isConnected && styles.connectBtnActive]}
            onPress={isConnected ? disconnect : connect}
            activeOpacity={0.7}
          >
            <Ionicons
              name={isConnected ? 'power' : 'power-outline'}
              size={16}
              color={isConnected ? Colors.danger : Colors.success}
            />
            <Text style={[styles.connectBtnText, isConnected && { color: Colors.danger }]}>
              {isConnected ? 'Desconectar' : 'Conectar'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Último Evento */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Ionicons name="pulse-outline" size={18} color={Colors.textSecondary} />
          <Text style={styles.cardTitle}>Último Evento</Text>
        </View>
        {lastEvent ? (
          <>
            <Text style={styles.cardMessage}>
              {lastEvent.payload.alert === 'TEST' ? 'Teste de Alarme' : 'Possível Queda Detectada'}
            </Text>
            <Text style={styles.cardSubtext}>
              {formatTimestamp(lastEvent.receivedAt)} · Confiança: {(lastEvent.payload.confidence * 100).toFixed(0)}%
            </Text>
          </>
        ) : (
          <>
            <Text style={styles.cardMessage}>Nenhum evento registrado</Text>
            <Text style={styles.cardSubtext}>O sistema está monitorando</Text>
          </>
        )}
      </View>

      {/* Estatísticas */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{todayCount}</Text>
          <Text style={styles.statLabel}>Alertas Hoje</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>{weekCount}</Text>
          <Text style={styles.statLabel}>Esta Semana</Text>
        </View>
      </View>

      {/* Botão de Teste */}
      <TouchableOpacity style={styles.testButton} onPress={testAlarm} activeOpacity={0.7}>
        <Ionicons name="notifications-outline" size={20} color={Colors.textPrimary} />
        <Text style={styles.testButtonText}>Testar Alarme</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    padding: 20,
    paddingTop: 60,
  },
  header: {
    fontSize: 28,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 24,
  },
  section: {
    marginBottom: 20,
  },
  sectionLabel: {
    fontSize: 13,
    color: Colors.textSecondary,
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  connectBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.successDim,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    gap: 6,
  },
  connectBtnActive: {
    backgroundColor: Colors.dangerDim,
  },
  connectBtnText: {
    color: Colors.success,
    fontSize: 13,
    fontWeight: '600',
  },
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 18,
    marginBottom: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 14,
    color: Colors.textSecondary,
    marginLeft: 8,
    fontWeight: '500',
  },
  cardMessage: {
    fontSize: 16,
    color: Colors.textPrimary,
    fontWeight: '600',
    marginBottom: 4,
  },
  cardSubtext: {
    fontSize: 13,
    color: Colors.textMuted,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  statCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 18,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 32,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
  testButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.surfaceLight,
    borderRadius: 14,
    paddingVertical: 16,
    gap: 8,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  testButtonText: {
    color: Colors.textPrimary,
    fontSize: 15,
    fontWeight: '600',
  },
});
