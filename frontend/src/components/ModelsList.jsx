import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Text,
  VStack,
  HStack,
  Badge,
  Button,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  Spinner,
  Alert,
  AlertIcon,
  Card,
  CardHeader,
  CardBody,
  useToast
} from '@chakra-ui/react';
import { DownloadIcon, ViewIcon, WarningIcon, DeleteIcon } from '@chakra-ui/icons';
import axios from 'axios';

const ModelsList = ({ onSelectModel }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stoppingModels, setStoppingModels] = useState([]);
  const [deletingModels, setDeletingModels] = useState([]);
  const toast = useToast();

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5000/api/trained-models');
      setModels(response.data.trained_models || []);
      setError(null);
    } catch (err) {
      console.error('Error fetching models:', err);
      setError('Failed to load trained models. Please check if the server is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
    
    // Refresh list every 20 seconds
    const interval = setInterval(() => {
      fetchModels();
    }, 20000);
    
    return () => clearInterval(interval);
  }, []);

  const handleDownload = async (modelId) => {
    try {
      // Trigger file download
      window.open(`http://localhost:5000/api/models/${modelId}/download`, '_blank');
      
      toast({
        title: 'Download started',
        description: 'Your model download has been initiated.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Download failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleStopTraining = async (modelId) => {
    if (stoppingModels.includes(modelId)) return;
    
    try {
      setStoppingModels(prev => [...prev, modelId]);
      
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
      
      // Refresh the list after a short delay
      setTimeout(fetchModels, 1000);
      
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
      setStoppingModels(prev => prev.filter(id => id !== modelId));
    }
  };

  const handleDeleteModel = async (modelId) => {
    if (deletingModels.includes(modelId)) return;
    
    if (!window.confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
      return;
    }
    
    try {
      setDeletingModels(prev => [...prev, modelId]);
      
      const response = await axios.delete(`http://localhost:5000/api/delete-model/${modelId}`);
      
      toast({
        title: 'Model deleted',
        description: response.data.message || 'The model has been successfully deleted',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      
      // Refresh the list
      fetchModels();
      
    } catch (error) {
      console.error('Error deleting model:', error);
      toast({
        title: 'Failed to delete model',
        description: error.response?.data?.error || error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setDeletingModels(prev => prev.filter(id => id !== modelId));
    }
  };

  const canStopTraining = (status) => {
    return ['initializing', 'preparing', 'starting', 'running', 'training'].includes(status);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'training':
      case 'running':
      case 'starting':
        return 'green';
      case 'completed':
        return 'blue';
      case 'error':
        return 'red';
      default:
        return 'gray';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Card variant="outline" width="100%" mb={4}>
      <CardHeader pb={2}>
        <HStack justify="space-between">
          <Heading size="md">Trained Models</Heading>
          <Button size="sm" onClick={fetchModels} isLoading={loading}>
            Refresh List
          </Button>
        </HStack>
      </CardHeader>
      <CardBody>
        {error && (
          <Alert status="error" mb={4}>
            <AlertIcon />
            {error}
          </Alert>
        )}

        {loading ? (
          <Box textAlign="center" py={4}>
            <Spinner />
            <Text mt={2}>Loading models...</Text>
          </Box>
        ) : models.length === 0 ? (
          <Alert status="info">
            <AlertIcon />
            No trained models found. Start training a model first.
          </Alert>
        ) : (
          <TableContainer>
            <Table variant="simple" size="sm">
              <Thead>
                <Tr>
                  <Th>Model ID</Th>
                  <Th>Status</Th>
                  <Th>Progress</Th>
                  <Th>Epochs</Th>
                  <Th>Metrics</Th>
                  <Th>File Size</Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {models.map((model) => (
                  <Tr key={model.id}>
                    <Td>{model.id.substring(0, 8)}...</Td>
                    <Td>
                      <Badge colorScheme={getStatusColor(model.status)}>
                        {model.status.toUpperCase()}
                      </Badge>
                    </Td>
                    <Td>{Math.round(model.progress)}%</Td>
                    <Td>{model.completed_epochs}/{model.epochs}</Td>
                    <Td>
                      {model.metrics && Object.keys(model.metrics).length > 0 ? (
                        <VStack align="start" spacing={0}>
                          {typeof model.metrics["metrics/mAP50(B)"] !== 'undefined' && (
                            <Text fontSize="xs">mAP@50: {Number(model.metrics["metrics/mAP50(B)"]).toFixed(3)}</Text>
                          )}
                          {typeof model.metrics["metrics/mAP50-95(B)"] !== 'undefined' && (
                            <Text fontSize="xs">mAP@50-95: {Number(model.metrics["metrics/mAP50-95(B)"]).toFixed(3)}</Text>
                          )}
                          {typeof model.metrics["metrics/precision(B)"] !== 'undefined' && (
                            <Text fontSize="xs">Precision: {Number(model.metrics["metrics/precision(B)"]).toFixed(3)}</Text>
                          )}
                          {typeof model.metrics["metrics/recall(B)"] !== 'undefined' && (
                            <Text fontSize="xs">Recall: {Number(model.metrics["metrics/recall(B)"]).toFixed(3)}</Text>
                          )}
                        </VStack>
                      ) : (
                        <Text fontSize="xs">No metrics</Text>
                      )}
                    </Td>
                    <Td>
                      {model.model_exists ? formatFileSize(model.file_size) : 'Not available'}
                    </Td>
                    <Td>
                      <HStack spacing={2}>
                        {model.model_exists && (
                          <Button 
                            size="xs" 
                            leftIcon={<DownloadIcon />} 
                            colorScheme="blue"
                            onClick={() => handleDownload(model.id)}
                          >
                            Download
                          </Button>
                        )}
                        {canStopTraining(model.status) && (
                          <Button 
                            size="xs" 
                            leftIcon={<WarningIcon />}
                            colorScheme="red"
                            isLoading={stoppingModels.includes(model.id)}
                            loadingText="Stopping"
                            onClick={() => handleStopTraining(model.id)}
                          >
                            Stop
                          </Button>
                        )}
                        <Button 
                          size="xs" 
                          leftIcon={<ViewIcon />} 
                          onClick={() => onSelectModel && onSelectModel(model.id)}
                        >
                          Select
                        </Button>
                        {model.status !== 'training' && model.status !== 'starting' && (
                          <Button 
                            size="xs" 
                            leftIcon={<DeleteIcon />}
                            colorScheme="red"
                            isLoading={deletingModels.includes(model.id)}
                            loadingText="Deleting"
                            onClick={() => handleDeleteModel(model.id)}
                          >
                            Delete
                          </Button>
                        )}
                      </HStack>
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </TableContainer>
        )}
      </CardBody>
    </Card>
  );
};

export default ModelsList; 