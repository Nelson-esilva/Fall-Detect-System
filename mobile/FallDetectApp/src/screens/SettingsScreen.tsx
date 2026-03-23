import React from 'react';
import { View, Text, StyleSheet, ScrollView, TextInput, Switch } from 'react-native';
import { Colors } from '../theme/colors';
import { useApp } from '../context/AppContext';

export function SettingsScreen() {
  const { settings, updateSettings, mqttStatus } = useApp();
  const isConnected = mqttStatus === 'connected';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.header}>Configurações</Text>

      {/* MQTT */}
      <Text style={styles.sectionLabel}>Conexão MQTT</Text>
      <View style={styles.card}>
        <View style={styles.field}>
          <Text style={styles.fieldLabel}>Endereço do Broker</Text>
          <TextInput
            style={[styles.input, isConnected && styles.inputDisabled]}
            value={settings.brokerHost}
            onChangeText={(v) => updateSettings({ brokerHost: v })}
            placeholder="192.168.1.100"
            placeholderTextColor={Colors.textMuted}
            autoCapitalize="none"
            editable={!isConnected}
          />
        </View>
        <View style={styles.divider} />
        <View style={styles.field}>
          <Text style={styles.fieldLabel}>Porta (WebSocket)</Text>
          <TextInput
            style={[styles.input, isConnected && styles.inputDisabled]}
            value={String(settings.brokerPort)}
            onChangeText={(v) => updateSettings({ brokerPort: parseInt(v, 10) || 0 })}
            placeholder="9001"
            placeholderTextColor={Colors.textMuted}
            keyboardType="numeric"
            editable={!isConnected}
          />
        </View>
        <View style={styles.divider} />
        <View style={styles.field}>
          <Text style={styles.fieldLabel}>Tópico</Text>
          <TextInput
            style={[styles.input, isConnected && styles.inputDisabled]}
            value={settings.topic}
            onChangeText={(v) => updateSettings({ topic: v })}
            placeholder="fall_detection/alerts"
            placeholderTextColor={Colors.textMuted}
            autoCapitalize="none"
            editable={!isConnected}
          />
        </View>
      </View>
      {isConnected && (
        <Text style={styles.hint}>Desconecte para alterar as configurações de conexão.</Text>
      )}

      {/* Emergência */}
      <Text style={styles.sectionLabel}>Emergência</Text>
      <View style={styles.card}>
        <View style={styles.field}>
          <Text style={styles.fieldLabel}>Número de Emergência</Text>
          <TextInput
            style={styles.input}
            value={settings.emergencyNumber}
            onChangeText={(v) => updateSettings({ emergencyNumber: v })}
            placeholder="192"
            placeholderTextColor={Colors.textMuted}
            keyboardType="phone-pad"
          />
        </View>
      </View>

      {/* Alarme */}
      <Text style={styles.sectionLabel}>Alarme</Text>
      <View style={styles.card}>
        <View style={styles.field}>
          <Text style={styles.fieldLabel}>Limiar de Confiança</Text>
          <TextInput
            style={styles.input}
            value={String(Math.round(settings.confidenceThreshold * 100))}
            onChangeText={(v) => {
              const num = parseInt(v, 10);
              if (!isNaN(num)) updateSettings({ confidenceThreshold: Math.min(100, Math.max(0, num)) / 100 });
            }}
            placeholder="70"
            placeholderTextColor={Colors.textMuted}
            keyboardType="numeric"
          />
          <Text style={styles.fieldSuffix}>%</Text>
        </View>
      </View>

      {/* Notificações */}
      <Text style={styles.sectionLabel}>Notificações</Text>
      <View style={styles.card}>
        <View style={styles.switchRow}>
          <Text style={styles.fieldLabel}>Notificações Push</Text>
          <Switch
            value={settings.notificationsEnabled}
            onValueChange={(v) => updateSettings({ notificationsEnabled: v })}
            trackColor={{ false: Colors.surfaceLight, true: Colors.primary }}
            thumbColor={Colors.textPrimary}
          />
        </View>
      </View>

      <View style={{ height: 40 }} />
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
  sectionLabel: {
    fontSize: 13,
    color: Colors.textSecondary,
    marginBottom: 8,
    marginTop: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 14,
    marginBottom: 4,
    overflow: 'hidden',
  },
  field: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  fieldLabel: {
    flex: 1,
    fontSize: 15,
    color: Colors.textPrimary,
  },
  fieldSuffix: {
    fontSize: 14,
    color: Colors.textSecondary,
    marginLeft: 4,
  },
  input: {
    color: Colors.textSecondary,
    fontSize: 15,
    textAlign: 'right',
    minWidth: 120,
  },
  inputDisabled: {
    opacity: 0.4,
  },
  divider: {
    height: 1,
    backgroundColor: Colors.divider,
    marginLeft: 16,
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  hint: {
    fontSize: 12,
    color: Colors.textMuted,
    marginBottom: 8,
    marginTop: 4,
    fontStyle: 'italic',
  },
});
