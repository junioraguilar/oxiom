import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Box, 
  Heading, 
  Text, 
  Divider,
  VStack,
  useToast,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Button,
} from '@chakra-ui/react';
import TrainingProgress from '../components/TrainingProgress';
import axios from 'axios';

const TrainingPage = () => {
  const toast = useToast();
  const [datasetId, setDatasetId] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [currentModelId, setCurrentModelId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [trainingSessions, setTrainingSessions] = useState([]);

  useEffect(() => {
    // Se houver um treinamento em andamento, redireciona para o monitor
    const checkAndRedirect = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/training-status');
        if (response.data.training_sessions && response.data.training_sessions.length > 0) {
          const activeSession = response.data.training_sessions.find(
            session => session.info.status === 'training' || 
                       session.info.status === 'starting' || 
                       session.info.status === 'running'
          );
          if (activeSession) {
            window.location.href = `/monitor/${activeSession.model_id}`;
            return;
          }
        }
      } catch (err) {
        // Ignora erros silenciosamente
      }
    };
    checkAndRedirect();
  }, []);

  useEffect(() => {
    // Fetch available datasets when component mounts
    const fetchDatasets = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:5000/api/datasets');
        setDatasets(response.data.datasets || []);
        
        // If datasets exist, select the first one
        if (response.data.datasets && response.data.datasets.length > 0) {
          setDatasetId(response.data.datasets[0].id);
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching datasets:', err);
        setError('Failed to load datasets. Please check if the server is running.');
      } finally {
        setLoading(false);
      }
    };

    // Fetch all training sessions
    const fetchTrainingSessions = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/training-status');
        if (response.data.training_sessions && response.data.training_sessions.length > 0) {
          setTrainingSessions(response.data.training_sessions);
          
          // Find any active training session
          const activeSession = response.data.training_sessions.find(
            session => session.info.status === 'training' || 
                      session.info.status === 'starting' || 
                      session.info.status === 'running'
          );
          
          if (activeSession) {
            setCurrentModelId(activeSession.model_id);
            toast({
              title: 'Training in progress',
              description: `Monitoring active training for model: ${activeSession.model_id}`,
              status: 'info',
              duration: 5000,
              isClosable: true,
            });
          }
        }
      } catch (err) {
        console.error('Error fetching training sessions:', err);
      }
    };

    fetchDatasets();
    fetchTrainingSessions();
    
    // Set an interval to refresh training sessions every 10 seconds
    const interval = setInterval(fetchTrainingSessions, 10000);
    
    // Clear the interval when component unmounts
    return () => clearInterval(interval);
  }, [toast]);

  const handleTrainingStart = (modelId) => {
    setCurrentModelId(modelId);
    // Refresh the training sessions list
    axios.get('http://localhost:5000/api/training-status')
      .then(response => {
        if (response.data.training_sessions) {
          setTrainingSessions(response.data.training_sessions);
        }
      })
      .catch(err => console.error('Error refreshing training sessions:', err));
  };
  
  const selectTrainingSession = (modelId) => {
    setCurrentModelId(modelId);
  };

  return (
    <Container maxW="container.xl" py={8}>
      <VStack spacing={8} align="stretch">
        <Box>
          <Heading as="h1" size="xl">YOLOv8 Model Training</Heading>
          <Text mt={2} color="gray.600">Configure and monitor your object detection model training</Text>
        </Box>

        <Divider />

        {error && (
          <Alert status="error">
            <AlertIcon />
            <AlertTitle>Error!</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {datasets.length === 0 && !loading && !error ? (
          <Alert status="warning">
            <AlertIcon />
            <AlertTitle>No datasets found!</AlertTitle>
            <AlertDescription>Please upload a dataset first through the data management page.</AlertDescription>
          </Alert>
        ) : (
          <>
            {trainingSessions.length > 0 && (
              <Box mt={4}>
                <Heading size="md" mb={3}>Available Training Sessions</Heading>
                <VStack spacing={2} align="stretch">
                  {trainingSessions.map(session => (
                    <Button 
                      key={session.model_id}
                      onClick={() => selectTrainingSession(session.model_id)}
                      colorScheme={currentModelId === session.model_id ? 'blue' : 'gray'}
                      justifyContent="space-between"
                      variant="outline"
                      size="sm"
                    >
                      <Text>Model: {session.model_id.substring(0, 8)}...</Text>
                      <Text>Status: {session.info.status}</Text>
                      <Text>Progress: {Math.round(session.info.progress * 100)}%</Text>
                    </Button>
                  ))}
                </VStack>
              </Box>
            )}

            {currentModelId && (
              <TrainingProgress 
                modelId={currentModelId} 
              />
            )}
          </>
        )}
      </VStack>
    </Container>
  );
};

export default TrainingPage; 