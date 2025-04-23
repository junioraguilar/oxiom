import React, { useState, useEffect } from 'react';
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
  useDisclosure,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay
} from '@chakra-ui/react';
import ModelsList from '../components/ModelsList';
import axios from 'axios';

const ModelsPage = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedModelInfo, setSelectedModelInfo] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const cancelRef = React.useRef();

  // Buscar informações do modelo selecionado
  useEffect(() => {
    if (selectedModel) {
      fetchModelInfo(selectedModel);
    }
  }, [selectedModel]);

  const fetchModelInfo = async (modelId) => {
    try {
      const response = await axios.get(`http://localhost:5000/api/training-status?model_id=${modelId}`);
      setSelectedModelInfo(response.data);
    } catch (error) {
      console.error('Error fetching model info:', error);
      setSelectedModelInfo(null);
    }
  };

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

  const handleDeleteModel = async () => {
    if (!selectedModel) return;
    
    try {
      setIsDeleting(true);
      
      const response = await axios.delete(`http://localhost:5000/api/models/${selectedModel}/delete`);
      
      toast({
        title: 'Model deleted',
        description: response.data.message || 'The model has been successfully deleted',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      
      // Limpar modelo selecionado e fechar diálogo
      setSelectedModel(null);
      setSelectedModelInfo(null);
      onClose();
      
      // Forçar atualização da lista de modelos (trigger re-render)
      setTimeout(() => {
        window.location.reload();
      }, 1000);
      
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
      setIsDeleting(false);
    }
  };

  // Verificar se o modelo pode ser excluído (não está em treinamento)
  const canDeleteModel = () => {
    if (!selectedModelInfo) return false;
    const status = selectedModelInfo.status;
    return status !== 'training' && status !== 'starting';
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
              {canDeleteModel() && (
                <Button 
                  colorScheme="red"
                  onClick={onOpen}
                >
                  Delete Model
                </Button>
              )}
            </HStack>
          </Box>
        )}
      </VStack>
      
      <AlertDialog
        isOpen={isOpen}
        leastDestructiveRef={cancelRef}
        onClose={onClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Model
            </AlertDialogHeader>

            <AlertDialogBody>
              Are you sure you want to delete this model? This action cannot be undone.
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onClose}>
                Cancel
              </Button>
              <Button 
                colorScheme="red" 
                onClick={handleDeleteModel} 
                ml={3}
                isLoading={isDeleting}
                loadingText="Deleting"
              >
                Delete
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </Container>
  );
};

export default ModelsPage; 