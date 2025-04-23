import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Heading, 
  Text, 
  Divider,
  VStack,
  HStack,
  Button,
  useToast,
  useDisclosure
} from '@chakra-ui/react';
import ModelsList from '../components/ModelsList';

const ModelsPage = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const toast = useToast();

  const handleSelectModel = (modelId) => {
    setSelectedModel(modelId);
    toast({
      title: 'Model selected',
      description: `Selected model: ${modelId}`,
      status: 'info',
      duration: 3000,
      isClosable: true,
    });
  };

  return (
    <Container maxW="container.xl" py={8}>
      <VStack spacing={8} align="stretch">
        <Box>
          <Heading as="h1" size="xl">YOLOv8 Models</Heading>
          <Text mt={2} color="gray.600">View, download and manage your trained models</Text>
        </Box>

        <Divider />
        
        <ModelsList onSelectModel={handleSelectModel} />
        
        {selectedModel && (
          <Box p={4} borderWidth="1px" borderRadius="lg">
            <Heading size="md" mb={4}>Selected Model: {selectedModel}</Heading>
            <Text mb={4}>
              You can now use this model for object detection. Go to the Test Model page to run detection on images.
            </Text>
            <HStack>
              <Button 
                colorScheme="blue"
                onClick={() => window.open(`http://localhost:5000/api/models/${selectedModel}/download`, '_blank')}
              >
                Download Model
              </Button>
              <Button 
                colorScheme="green"
                onClick={() => window.location.href = '/test'}
              >
                Test Model
              </Button>
            </HStack>
          </Box>
        )}
      </VStack>
    </Container>
  );
};

export default ModelsPage; 