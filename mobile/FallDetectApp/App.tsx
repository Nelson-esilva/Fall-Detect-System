import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from './src/theme/colors';
import { AppProvider, useApp } from './src/context/AppContext';
import { DashboardScreen } from './src/screens/DashboardScreen';
import { AlarmScreen } from './src/screens/AlarmScreen';
import { HistoryScreen } from './src/screens/HistoryScreen';
import { SettingsScreen } from './src/screens/SettingsScreen';

const Tab = createBottomTabNavigator();

const DarkTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    background: Colors.background,
    card: Colors.tabBarBackground,
    text: Colors.textPrimary,
    border: Colors.border,
    primary: Colors.primary,
  },
};

const TAB_ICONS: Record<string, { focused: keyof typeof Ionicons.glyphMap; unfocused: keyof typeof Ionicons.glyphMap }> = {
  Dashboard: { focused: 'home', unfocused: 'home-outline' },
  Alarme: { focused: 'notifications', unfocused: 'notifications-outline' },
  'Histórico': { focused: 'time', unfocused: 'time-outline' },
  'Configurações': { focused: 'settings', unfocused: 'settings-outline' },
};

function AppNavigator() {
  const { navigationRef, alarmActive } = useApp();

  return (
    <NavigationContainer theme={DarkTheme} ref={navigationRef}>
      <StatusBar style="light" />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerShown: false,
          tabBarIcon: ({ focused, color, size }) => {
            const icons = TAB_ICONS[route.name];
            const iconName = focused ? icons.focused : icons.unfocused;
            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: Colors.tabBarActive,
          tabBarInactiveTintColor: Colors.tabBarInactive,
          tabBarStyle: {
            backgroundColor: Colors.tabBarBackground,
            borderTopColor: Colors.border,
            borderTopWidth: 1,
            height: 60,
            paddingBottom: 8,
            paddingTop: 4,
          },
          tabBarLabelStyle: {
            fontSize: 11,
            fontWeight: '600',
          },
        })}
      >
        <Tab.Screen name="Dashboard" component={DashboardScreen} />
        <Tab.Screen
          name="Alarme"
          component={AlarmScreen}
          options={{
            tabBarBadge: alarmActive ? '!' : undefined,
            tabBarBadgeStyle: { backgroundColor: Colors.danger },
          }}
        />
        <Tab.Screen name="Histórico" component={HistoryScreen} />
        <Tab.Screen name="Configurações" component={SettingsScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppNavigator />
    </AppProvider>
  );
}
