import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Heading, 
  Text, 
  Progress, 
  Stat, 
  StatLabel, 
  StatNumber, 
  StatGroup, 
  Badge, 
  VStack,
  HStack,
  Card,
  CardHeader,
  CardBody,
  Divider,
  SimpleGrid,
  Alert,
  AlertIcon,
  Spinner,
  Button,
  useToast
} from '@chakra-ui/react';
import { io } from '../../node_modules/socket.io-client/dist/socket.io.esm.min.js';
import { WarningIcon } from '@chakra-ui/icons';
import axios from 'axios';
import MetricsPanel from './MetricsPanel';

const TrainingProgress = ({ modelId }) => {
  const [trainingInfo, setTrainingInfo] = useState({
    status: 'initializing',
    progress: 0,
    current_epoch: 0,
    total_epochs: 0,
    metrics: {},
    error_message: null
  });
  const [connected, setConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [stoppingTraining, setStoppingTraining] = useState(false);
  const [metricsHistory, setMetricsHistory] = useState({});
  const toast = useToast();

  useEffect(() => {
    // Connect to the Socket.IO server with automatic reconnection
    const socket = io('http://localhost:5000', {
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      timeout: 10000
    });

    socket.on('connect', () => {
      console.log('Connected to Socket.IO server');
      setConnected(true);
      setConnectionError(null);
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from Socket.IO server');
      setConnected(false);
    });

    socket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      setConnectionError('Failed to connect to server. Please check if the backend is running.');
    });

    // Listen for training updates
    socket.on('training_update', (data) => {
      console.log('Received training update:', data);
      
      if (data.model_id === modelId && data.info) {
        // Ensure progress is within the 0-1 range
        const info = {...data.info};
        
        // Make sure progress is a decimal between 0 and 1
        if (typeof info.progress === 'number') {
          if (info.progress > 1) {
            // If progress is sent as a percentage (e.g. 4 instead of 0.04), convert it
            info.progress = info.progress / 100;
          }
          
          // Safety check to keep progress in valid range
          info.progress = Math.max(0, Math.min(1, info.progress));
          console.log(`Progress normalized: ${info.progress}`);
        }
        
        console.log("Metrics received:", info.metrics);
        setTrainingInfo(info);

        // Atualiza histórico das métricas
        if (info.metrics && info.current_epoch) {
          setMetricsHistory(prev => {
            const updated = { ...prev };
            Object.entries(info.metrics).forEach(([key, value]) => {
              if (!updated[key]) updated[key] = [];
              // Evita duplicatas para a mesma epoch
              if (!updated[key].some(item => item.epoch === info.current_epoch)) {
                updated[key] = [...updated[key], { epoch: info.current_epoch, value: value }];
                // Mantém só os últimos 30 valores
                if (updated[key].length > 30) updated[key] = updated[key].slice(-30);
              }
            });
            return updated;
          });
        }
      }
    });

    // Initial request for training information
    socket.emit('get_training_info', { model_id: modelId });

    // Clean up on unmount
    return () => {
      socket.disconnect();
    };
  }, [modelId]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'training':
        return 'green';
      case 'completed':
        return 'blue';
      case 'failed':
        return 'red';
      case 'initializing':
        return 'orange';
      case 'stopped':
        return 'gray';
      default:
        return 'gray';
    }
  };

  const renderMetrics = () => {
    if (!trainingInfo.metrics || Object.keys(trainingInfo.metrics).length === 0) {
      return (
        <Text>No metrics available yet</Text>
      );
    }

    // Remove epoch_percent das métricas exibidas
    const metricsArray = Object.entries(trainingInfo.metrics).filter(([key]) => key !== 'epoch_percent');
    
    return (
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4}>
        {metricsArray.map(([key, value]) => (
          <Stat key={key} p={2} shadow="sm" border="1px" borderColor="gray.200" borderRadius="md">
            <StatLabel>{key}</StatLabel>
            <StatNumber>{typeof value === 'number' ? value.toFixed(4) : value}</StatNumber>
          </Stat>
        ))}
      </SimpleGrid>
    );
  };

  // Add a new function to stop training
  const handleStopTraining = async () => {
    if (!modelId) return;
    
    try {
      setStoppingTraining(true);
      
      const response = await axios.post('http://localhost:5000/api/training/stop', {
        model_id: modelId
      });
      
      toast({
        title: 'Training stop requested',
        description: response.data.message || 'Training will stop at the end of the current epoch',
        status: 'info',
        duration: 5000,
        isClosable: true,
      });
      
    } catch (error) {
      console.error('Error stopping training:', error);
      toast({
        title: 'Failed to stop training',
        description: error.response?.data?.error || error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setStoppingTraining(false);
    }
  };

  // Check if training is active and can be stopped
  const canStopTraining = ['initializing', 'preparing', 'starting', 'running', 'training'].includes(trainingInfo.status);

  return (
    <Card variant="outline" width="100%">
      <CardHeader>
        <HStack justify="space-between">
          <Heading size="md">Training Progress</Heading>
          <HStack>
            {!connected && <Spinner size="sm" />}
            <Badge 
              colorScheme={connected ? 'green' : 'red'}
            >
              {connected ? 'Connected' : 'Disconnected'}
            </Badge>
            <Badge 
              colorScheme={getStatusColor(trainingInfo.status)}
            >
              {trainingInfo.status?.toUpperCase()}
            </Badge>
            {canStopTraining && (
              <Button
                size="sm"
                colorScheme="red"
                leftIcon={<WarningIcon />}
                onClick={handleStopTraining}
                isLoading={stoppingTraining}
                loadingText="Stopping"
              >
                Stop Training
              </Button>
            )}
          </HStack>
        </HStack>
      </CardHeader>
      <CardBody>
        <MetricsPanel metricsHistory={metricsHistory} currentMetrics={trainingInfo.metrics} />
        <VStack spacing={4} align="stretch">
          {connectionError && (
            <Alert status="error">
              <AlertIcon />
              {connectionError}
            </Alert>
          )}
          
          {trainingInfo.error_message && (
            <Alert status="error">
              <AlertIcon />
              {trainingInfo.error_message}
            </Alert>
          )}

          <Box>
            <HStack justify="space-between">
              <Text fontWeight="bold">Progress</Text>
            </HStack>
            <Progress 
              value={trainingInfo.progress * 100} 
              size="lg" 
              colorScheme={getStatusColor(trainingInfo.status)}
              hasStripe={trainingInfo.status === 'training'}
              isAnimated={trainingInfo.status === 'training'}
            />
          </Box>

          <HStack>
            <Stat>
              <StatLabel>Current Epoch</StatLabel>
              <StatNumber>{trainingInfo.current_epoch}</StatNumber>
            </Stat>
            <Stat>
              <StatLabel>Total Epochs</StatLabel>
              <StatNumber>{trainingInfo.total_epochs}</StatNumber>
            </Stat>
          </HStack>

          <Divider />
          
          <Heading size="sm" mb={2}>Training Metrics</Heading>
          {renderMetrics()}
        </VStack>
      </CardBody>
    </Card>
  );
};

export default TrainingProgress; 