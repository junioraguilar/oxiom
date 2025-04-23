import { useState } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  Button,
  FormControl,
  FormLabel,
  FormHelperText,
  Input,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Progress,
  Box,
  Text,
  useToast
} from '@chakra-ui/react';
import api from '../api/axios';

interface DatasetUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete: () => void;
}

const DatasetUploadModal = ({ isOpen, onClose, onUploadComplete }: DatasetUploadModalProps) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  
  const toast = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      
      // Verificar se o arquivo é um ZIP
      if (!file.name.toLowerCase().endsWith('.zip')) {
        toast({
          title: 'Formato inválido',
          description: 'Por favor, selecione um arquivo ZIP',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
        return;
      }
      
      // Verificar o tamanho do arquivo (limite de 900MB para segurança)
      const maxSize = 900 * 1024 * 1024; // 900MB em bytes
      if (file.size > maxSize) {
        toast({
          title: 'Arquivo muito grande',
          description: 'O tamanho máximo do arquivo é 900MB',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
        return;
      }
      
      setSelectedFile(file);
      setError('');
    }
  };

  const handleUploadDataset = async () => {
    if (!selectedFile) {
      toast({
        title: 'Nenhum arquivo selecionado',
        description: 'Por favor, selecione um arquivo de dataset para upload',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setIsUploading(true);
    setUploadProgress(0);
    setError('');

    try {
      const response = await api.post('/api/upload-dataset', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100)
          );
          setUploadProgress(percentCompleted);
        },
      });

      toast({
        title: 'Upload bem-sucedido',
        description: 'Dataset foi enviado com sucesso',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      
      // Reset form and close modal
      setSelectedFile(null);
      setIsUploading(false);
      onUploadComplete();
      onClose();
    } catch (error: any) {
      console.error('Upload falhou:', error);
      
      let errorMsg = 'Falha ao enviar dataset';
      if (error.code === 'ECONNABORTED') {
        errorMsg = 'O upload expirou. O arquivo pode ser muito grande.';
      } else if (error.response) {
        errorMsg = error.response.data?.error || errorMsg;
      }
      
      setError(errorMsg);
      toast({
        title: 'Falha no upload',
        description: errorMsg,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleClose = () => {
    if (!isUploading) {
      setSelectedFile(null);
      setError('');
      onClose();
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} size="md">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Upload de Dataset</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <FormControl>
            <FormLabel>Dataset (arquivo ZIP)</FormLabel>
            <Input
              type="file"
              accept=".zip"
              onChange={handleFileChange}
              disabled={isUploading}
              p={1}
            />
            <FormHelperText>
              Selecione um arquivo ZIP contendo seu dataset formatado para YOLO
            </FormHelperText>
          </FormControl>

          {error && (
            <Alert status="error" mt={4}>
              <AlertIcon />
              <AlertTitle mr={2}>Erro!</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {isUploading && (
            <Box mt={4}>
              <Progress value={uploadProgress} size="sm" colorScheme="blue" />
              <FormControl>
                <FormHelperText textAlign="center">{uploadProgress}% completo</FormHelperText>
              </FormControl>
            </Box>
          )}
        </ModalBody>

        <ModalFooter>
          <Button 
            colorScheme="blue" 
            mr={3} 
            onClick={handleUploadDataset}
            isLoading={isUploading}
            loadingText="Enviando..."
            disabled={!selectedFile || isUploading}
          >
            Upload
          </Button>
          <Button variant="ghost" onClick={handleClose} disabled={isUploading}>
            Cancelar
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default DatasetUploadModal; 