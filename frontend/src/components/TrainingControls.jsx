import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  FormControl, 
  FormLabel, 
  NumberInput, 
  NumberInputField, 
  NumberInputStepper, 
  NumberIncrementStepper, 
  NumberDecrementStepper,
  VStack,
  HStack,
  Divider,
  Heading,
  Text,
  useToast,
  Card,
  CardHeader,
  CardBody
} from '@chakra-ui/react';
import axios from 'axios';

const TrainingControls = ({ datasetId, onTrainingStart }) => {
  const toast = useToast();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    epochs: 50,
    batchSize: 16,
    imgSize: 640
  });

  const handleChange = (field, value) => {
    setFormData({
      ...formData,
      [field]: value
    });
  };

  const startTraining = async () => {
    if (!datasetId) {
      toast({
        title: 'No dataset selected',
        description: 'Please upload a dataset before starting training',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/api/train', {
        dataset_id: datasetId,
        epochs: formData.epochs,
        batch_size: formData.batchSize,
        img_size: formData.imgSize
      });

      toast({
        title: 'Training started',
        description: `Model ID: ${response.data.model_id}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });

      if (onTrainingStart) {
        onTrainingStart(response.data.model_id);
      }
    } catch (error) {
      console.error('Error starting training:', error);
      toast({
        title: 'Error starting training',
        description: error.response?.data?.error || error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card variant="filled" my={4}>
      <CardHeader pb={0}>
        <Heading size="md">Training Configuration</Heading>
      </CardHeader>
      <CardBody>
        <VStack spacing={4} align="stretch">
          <FormControl>
            <FormLabel>Epochs</FormLabel>
            <NumberInput 
              min={1} 
              max={1000} 
              value={formData.epochs}
              onChange={(_, value) => handleChange('epochs', value)}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <Text fontSize="xs" color="gray.500">
              Number of training cycles through the dataset
            </Text>
          </FormControl>

          <FormControl>
            <FormLabel>Batch Size</FormLabel>
            <NumberInput 
              min={1} 
              max={128} 
              value={formData.batchSize}
              onChange={(_, value) => handleChange('batchSize', value)}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <Text fontSize="xs" color="gray.500">
              Number of images processed together (lower for less GPU memory)
            </Text>
          </FormControl>

          <FormControl>
            <FormLabel>Image Size</FormLabel>
            <NumberInput 
              min={320} 
              max={1280} 
              step={32}
              value={formData.imgSize}
              onChange={(_, value) => handleChange('imgSize', value)}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <Text fontSize="xs" color="gray.500">
              Input image size for training (larger = more accuracy but slower)
            </Text>
          </FormControl>

          <Divider />

          <Button 
            colorScheme="blue" 
            isLoading={loading}
            onClick={startTraining}
            isDisabled={!datasetId}
          >
            Start Training
          </Button>
          
          {!datasetId && (
            <Text fontSize="sm" color="orange.500">
              Please upload a dataset first
            </Text>
          )}
        </VStack>
      </CardBody>
    </Card>
  );
};

export default TrainingControls; 